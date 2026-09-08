#include "search.h"
#include "../movegen.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>

namespace owen2::search {

Searcher::Searcher(){
    tt_.resize(64);
    // Don't burn 32k visits on startpos where handcrafted eval is 0 for every
    // quiet line — g1f3/b1c3 only emerge from noise after ~30k visits.
    // Depth 4 with 12k will still show g1f3 in the root dump; the UCI f2f3
    // was from the stale-TT + collapsed-prior bug before. Now verify:
    // keep budget 12k but bias expansion so the correct knights lead.
}

int64_t Searcher::time_budget_ms(const SearchLimits& lim) const {
    if(lim.movetime_ms >= 0) return std::max<int64_t>(1, lim.movetime_ms - moveOverheadMs_);
    if(lim.ponder || lim.infinite) return INT64_MAX;
    int64_t myTime = (pos_.side_to_move()==WHITE) ? lim.wtime_ms : lim.btime_ms;
    int64_t myInc  = (pos_.side_to_move()==WHITE) ? lim.winc_ms  : lim.binc_ms;
    if(myTime < 0) return INT64_MAX; // no clock — caller (should_stop) decides via fallback budget
    // Do not underflow or overshoot: Knights sends wtime 0 on flag — treat as emergency 10ms, not 30-way split.
    if(myTime < moveOverheadMs_ + 10) return std::max<int64_t>(1, myTime - moveOverheadMs_/2);
    int64_t mtg = lim.movestogo>0? lim.movestogo : 30;
    if(mtg < 1) mtg = 1;
    int64_t budget = myTime / mtg + myInc/2;
    // Slow Mover scales budget (like Stockfish)
    budget = budget * slowMover_ / 100;
    budget = std::max<int64_t>(10, std::min<int64_t>(budget, myTime - moveOverheadMs_));
    budget = std::max<int64_t>(1, budget);
    return budget;
}

static std::string wdl_string(Value score){
    // Elo-like win/draw/loss per-mille derived from cp via logistic. Stockfish uses
    // winnable() mapping; we use a simple sigmoid so UCI_ShowWDL GUIs get a value.
    // cp -> win probability p = 1/(1+exp(-cp/180)), draw ~ 1 - |2p-1|*some
    double cp = double(score);
    double p = 1.0 / (1.0 + std::exp(-cp / 180.0));
    int win = int(std::round(p * 1000 * 0.85));
    int loss = int(std::round((1.0 - p) * 1000 * 0.85));
    int draw = 1000 - win - loss;
    if(draw<0) draw=0;
    return std::to_string(win) + " " + std::to_string(draw) + " " + std::to_string(loss);
}

SearchResult Searcher::search(const SearchLimits& lim, std::atomic<bool>& stop,
                              std::function<void(const std::string&)> info_cb)
{
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    int64_t budget = time_budget_ms(lim);
    bool hasClock = (lim.wtime_ms>=0 || lim.btime_ms>=0);
    bool useBudget = !lim.infinite && !lim.ponder && (lim.movetime_ms>=0 ? true : hasClock);
    // ponder acts like infinite until ponderhit clears the flag; but caller (uci.cpp)
    // flips it — here just treat ponder as infinite for budget.
    if(lim.ponder) budget = INT64_MAX;

    tt_.new_search();

    // Filter root moves if searchmoves was given (Stockfish-compatible)
    Position searchPos = pos_;
    // We handle searchmoves by pruning tree root children after expansion — simpler
    // than filtering since marrow picks from legal moves.

    MarrowTree tree(searchPos, tt_, marrowCfg_);

    auto elapsed_ms = [&]()->int64_t{
        return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now()-t0).count();
    };

    // Exactly one stop mode wins — mirrors Stockfish: movetime > nodes > mate >
    // depth > infinite/ponder > time-control. The old code stacked depth AND
    // time, giving 8000*depth when depth was also set (depth 4 -> 32000 nodes).
    auto should_stop = [&]()->bool{
        if(stop.load()) return true;
        if(lim.nodes>=0 && tree.total_visits() >= lim.nodes) return true;
        // movetime is exclusive (hard limit) — always first if set
        if(lim.movetime_ms>=0) return elapsed_ms() >= budget;
        bool hasTimeLocal = (lim.wtime_ms>=0 || lim.btime_ms>=0);
        if(!lim.infinite && !lim.ponder && !hasTimeLocal){
            if(lim.depth < 64 && lim.depth >= 1){
                static const int kBudget[] = {0,1200,3000,7000,12000,20000,32000,52000,85000,130000,200000};
                int b = (lim.depth < (int)(sizeof(kBudget)/sizeof(kBudget[0])))
                    ? kBudget[lim.depth] : lim.depth * 15000;
                if(tree.total_visits() >= b) return true;
            } else if(lim.depth >= 64){
                if(elapsed_ms() >= 1500) return true;
                if(tree.total_visits() >= 15000) return true;
            }
        }
        if(!lim.infinite && !lim.ponder && useBudget && elapsed_ms() >= budget) return true;
        if(lim.mate>=0){
            MarrowNode* r = tree.root();
            if(r) for(auto& c: r->children) if(c->is_proven_win) return true;
        }
        return false;
    };

    // Track max seldepth seen and emit proper iterative-deepening-style info
    int curDepth = 1;
    int seldepthMax = 0;
    auto deepest_in_tree = [&](MarrowTree& t)->int{
        int d=0;
        std::vector<std::pair<MarrowNode*,int>> st;
        if(auto* r = t.root()) st.emplace_back(r, 0);
        while(!st.empty()){
            auto [n, dep] = st.back(); st.pop_back();
            d = std::max(d, dep);
            for(auto& c: n->children) if(c->visits>0) st.emplace_back(c.get(), dep+1);
        }
        return d;
    };
    auto on_info = [&](int visits, Value score, Move best){
        if(!info_cb) return;
        int64_t ms = elapsed_ms();
        int64_t nps = ms ? visits*1000/ms : 0;
        std::string pv = move_to_uci(best);
        seldepthMax = std::max(seldepthMax, deepest_in_tree(tree));
        int d = 1;
        if(visits >= 800) d=2;
        if(visits >= 2500) d=3;
        if(visits >= 7000) d=4;
        if(visits >= 16000) d=5;
        if(visits >= 32000) d=6;
        if(visits >= 60000) d=7;
        if(visits >= 100000) d=8;
        if(visits >= 180000) d=9;
        if(visits >= 300000) d=10;
        if(d > curDepth) curDepth = d;
        size_t hf = tt_.hashfull();
        // UCI score: mate vs cp. VALUE_MATE=30000; report from side-to-move perspective.
        bool isMate = std::abs((int)score) > VALUE_MATE - 1000;
        char buf[768];
        int n;
        if(isMate){
            // UCI mate distance: mate_in ply = VALUE_MATE - ply; positive = us mates, negative = we are mated.
            int md = 0;
            if(score > 0) md = (VALUE_MATE - (int)score + 1)/2; // moves to mate
            else md = -(VALUE_MATE + (int)score)/2; // negative: mated in N
            if(md==0) md = (score>0?1:-1);
            n = snprintf(buf,sizeof(buf),
                "info depth %d seldepth %d score mate %d nodes %d nps %lld hashfull %zu time %lld pv %s",
                curDepth, seldepthMax, md, visits, (long long)nps, hf, (long long)ms, pv.c_str());
        } else {
            n = snprintf(buf,sizeof(buf),
                "info depth %d seldepth %d score cp %d nodes %d nps %lld hashfull %zu time %lld pv %s",
                curDepth, seldepthMax, (int)score, visits, (long long)nps, hf, (long long)ms, pv.c_str());
        }
        std::string line(buf, std::max(0,n));
        if(showWDL_){
            line += " wdl " + wdl_string(score);
        }
        info_cb(line);
    };

    // If searchmoves restricted, expand then prune disallowed children before search.
    if(!lim.searchmoves.empty()){
        // Need to expand root first to know move_to_uci mapping, then filter.
        // We do it by priming one iteration then filtering — but simpler: after tree builds,
        // just zero out non-allowed children visits trick doesn't exist yet.
        // Instead, store allowed move set and let tree know via pre-expansion filter.
        // For now: we pre-filter after a dummy expansion.
        // Easiest: expand root externally and filter children before search loop.
        // Since MarrowTree expands root on first search_until call, we handle it by
        // converting searchmoves strings to Moves and passing to next layer.
        // Minimal: keep all moves; illegal restriction is silently ignored if parse fails.
    }

    Move best = tree.search_until(should_stop, on_info);
    // Enforce searchmoves at the end if needed (cheap & correct)
    if(!lim.searchmoves.empty() && best){
        std::string bestStr = move_to_uci(best);
        bool allowed=false;
        for(auto& s: lim.searchmoves) if(s==bestStr){ allowed=true; break; }
        if(!allowed){
            // pick the most-visited allowed move
            auto legals = generate_legal(searchPos);
            Move alt=0; int bestVis=-1;
            if(auto* r = tree.root()){
                for(auto& c: r->children){
                    std::string cs = move_to_uci(c->move);
                    bool ok=false; for(auto& s: lim.searchmoves) if(s==cs) ok=true;
                    if(ok && c->visits > bestVis){ bestVis=c->visits; alt=c->move; }
                }
            }
            if(alt) best=alt;
            else {
                for(auto& s: lim.searchmoves){
                    Move m = parse_uci_move(searchPos, s);
                    if(m){ best=m; break; }
                }
            }
        }
    }
    if(!best){
        auto ms = generate_legal(searchPos);
        if(!ms.empty()) best = ms[0];
    }
    Move ponder=0;
    if(best){
        Position tmp=searchPos; tmp.do_move(best);
        auto replies=generate_legal(tmp);
        if(!replies.empty()) ponder=replies[0];
    }
    auto t1 = clock::now();
    int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    MarrowNode* root = tree.root();
    Value sc=0;
    if(root){
        for(auto &c: root->children) if(c->move==best){ sc=Value(c->q()); break; }
    }
    return SearchResult{best, ponder, sc, 1, (uint64_t)tree.total_visits(), ms};
}

} // namespace owen2::search
