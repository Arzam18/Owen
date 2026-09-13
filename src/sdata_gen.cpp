#include "position.h"
#include "movegen.h"
#include "bitboard.h"
#include "search/search.h"
#include "nnue/network.h"
#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <filesystem>

#pragma pack(push,1)
struct SDataRecord {
    uint8_t board[64];
    uint8_t stm;
    int16_t eval;
    uint8_t result;
    uint8_t ply;
    uint8_t castling; // bit0 K, bit1 Q, bit2 k, bit3 q
    uint8_t ep;       // 0..63 square, 64 = none
};
#pragma pack(pop)
static_assert(sizeof(SDataRecord)==71, "pack broken");

static void play_games(int thread_id, int games, int movetime_ms, int depth,
                       const std::string& net_path, const std::string& out_path,
                       std::atomic<int>& done, std::atomic<uint64_t>& positions,
                       std::mutex& io_mtx)
{
    owen2::search::Searcher searcher;
    searcher.set_tt_size(4); // small per-thread
    if(!net_path.empty()){
        std::lock_guard<std::mutex> lk(io_mtx);
        if(owen2::nnue::g_network.load(net_path))
            std::cout << "[t" << thread_id << "] loaded net " << net_path << "\n";
    }

    std::mt19937 rng(42 + thread_id * 1009);
    std::string part = out_path + ".part" + std::to_string(thread_id);
    std::ofstream f(part, std::ios::binary);
    if(!f){ std::cerr << "cannot open " << part << "\n"; return; }

    for(int g=0; g<games; ++g){
        owen2::Position pos;
        pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        searcher.set_position(pos);
        searcher.new_game();

        std::vector<SDataRecord> game;
        game.reserve(120);

        // small opening variety: 2 random plies
        for(int r=0; r<2; ++r){
            auto ms = owen2::generate_legal(pos);
            if(ms.empty()) break;
            owen2::Move rm = ms[rng() % ms.size()];
            pos.do_move(rm);
            searcher.set_position(pos);
        }

        for(int ply=0; ply<300; ++ply){
            auto moves = owen2::generate_legal(pos);
            if(moves.empty() || pos.is_draw()) break;

            owen2::search::SearchLimits lim;
            if(depth > 0) lim.depth = depth;
            else lim.movetime_ms = movetime_ms;

            std::atomic<bool> stop{false};
            auto res = searcher.search(lim, stop);
            owen2::Move m = res.bestMove;
            if(!m) m = moves[rng() % moves.size()];

            // NO_PIECE is 12 — distill.py expects 12 for empty, but C++ Piece
            // is an int enum. Cast is correct; the bug was that the struct
            // was zero-initialised with {} which sets empty squares to 0
            // (W_PAWN) inside make_piece histories before set_fen runs.
            // We explicitly skip uninitialized Position copies and validate.
            if(pos.piece_on(4)!=owen2::W_KING && pos.piece_on(4)!=owen2::B_KING){
                // fallback: skip records where king is missing — indicates
                // uninitialized Position copy in thread (race on g_network)
                continue;
            }
            SDataRecord r{};
            // Ensure empty squares are 12, not 0: piece_on returns 12 for empty,
            // but zero-init of SDataRecord would be 0. Overwrite all 64 explicitly.
            for(int s=0;s<64;++s){
                owen2::Piece pc = pos.piece_on(s);
                r.board[s]=(uint8_t)pc; // 0..11 piece, 12 empty
            }
            r.stm = (uint8_t)pos.side_to_move();
            r.eval = (int16_t)std::clamp<int>(res.score, -15000, 15000);
            r.ply = (uint8_t)std::min(ply, 255);
            r.castling = (uint8_t)pos.castling_rights();
            r.ep = (uint8_t)(pos.ep_square() >= 64 ? 64 : pos.ep_square());
            game.push_back(r);

            pos.do_move(m);
            searcher.set_position(pos);
        }

        int finalResult=1;
        auto ms = owen2::generate_legal(pos);
        if(ms.empty()){
            finalResult = pos.in_check() ? 0 : 1;
        } else if(pos.is_draw()) finalResult=1;
        else finalResult=1; // adjudicate long games as draw

        for(size_t i=0;i<game.size();++i){
            int dist = (int)game.size() - (int)i;
            int resFromStm;
            if(finalResult==1) resFromStm=1;
            else resFromStm = (dist%2==1) ? 2 : 0;
            game[i].result = (uint8_t)resFromStm;
            f.write((char*)&game[i], sizeof(SDataRecord));
        }
        positions += game.size();
        int d = ++done;
        if(d % 500 == 0 || g == games-1){
            std::lock_guard<std::mutex> lk(io_mtx);
            std::cout << "[" << d << "] games, " << positions.load() << " positions\n";
        }
    }
}

int main(int argc, char** argv){
    std::string out="data/sdata.bin";
    std::string net="";
    int games=1000;
    int threads=1;
    int movetime_ms=15;
    int depth=-1;

    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto need=[&](std::string &dst){ if(i+1<argc) dst=argv[++i]; };
        if(a=="--out") need(out);
        else if(a=="--games" && i+1<argc) games=std::stoi(argv[++i]);
        else if(a=="--threads" && i+1<argc) threads=std::stoi(argv[++i]);
        else if(a=="--movetime" && i+1<argc) movetime_ms=std::stoi(argv[++i]);
        else if(a=="--depth" && i+1<argc) depth=std::stoi(argv[++i]);
        else if(a=="--net" && i+1<argc) net=argv[++i];
        else if(a=="--help" || a=="-h"){
            std::cout << "Usage: owen2-sdata --games N --out file.bin [--threads T] [--movetime MS|--depth D] [--net file.o2nn]\n";
            return 0;
        }
    }
    owen2::init_attacks(); owen2::Position::init_zobrist();
    std::filesystem::create_directories(std::filesystem::path(out).parent_path());

    if(threads < 1) threads=1;
    if(threads > 64) threads=64;
    int per = games / threads;
    int rem = games % threads;

    std::cout << "Owen2 sdata: " << games << " games, " << threads << " threads, "
              << (depth>0 ? ("depth "+std::to_string(depth)) : ("movetime "+std::to_string(movetime_ms)+"ms"))
              << (net.empty()?" (handcrafted)":" net="+net) << "\n";

    std::atomic<int> done{0};
    std::atomic<uint64_t> positions{0};
    std::mutex io_mtx;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> pool;
    for(int t=0; t<threads; ++t){
        int g = per + (t < rem ? 1 : 0);
        if(g==0) continue;
        pool.emplace_back(play_games, t, g, movetime_ms, depth, net, out,
                          std::ref(done), std::ref(positions), std::ref(io_mtx));
    }
    for(auto &th: pool) th.join();

    // merge parts
    {
        std::ofstream fout(out, std::ios::binary);
        std::vector<char> buf(1<<20);
        for(int t=0; t<threads; ++t){
            std::string part = out + ".part" + std::to_string(t);
            if(!std::filesystem::exists(part)) continue;
            std::ifstream fin(part, std::ios::binary);
            while(fin){
                fin.read(buf.data(), buf.size());
                auto n = fin.gcount();
                if(n>0) fout.write(buf.data(), n);
            }
            std::filesystem::remove(part);
        }
    }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count();
    uint64_t posN = positions.load();
    double gps = ms ? (done.load()*1000.0/ms) : 0;
    std::cout << "Done: " << done.load() << " games, " << posN << " positions in " << ms/1000.0 << "s"
              << " (" << gps << " games/s, ~" << (posN*1000/ms) << " pos/s)\n";
    std::cout << "Wrote " << out << " (" << std::filesystem::file_size(out) << " bytes)\n";
    // estimate
    double needStockfish = 3e9; // positions
    std::cout << "Progress to ~Beat Stockfish ballpark (~1B pos): " << (posN/1e9*100) << "% this run, "
              << (posN>0? (1e9/posN):0) << "x more runs to 1B\n";
    return 0;
}
