#include "movegen.h"
#include "bitboard.h"
#include <cassert>

namespace owen2 {

static void add_move(std::vector<Move>& out, Square from, Square to, int flags, PieceType promo=QUEEN){
    if(flags & MoveFlag::PROMO) out.push_back(make_move(from,to,flags,promo));
    else out.push_back(make_move(from,to,flags));
}

std::vector<Move> generate_pseudo_legal(const Position& pos){
    std::vector<Move> moves; moves.reserve(64);
    Color us = pos.side_to_move(), them = ~us;
    Bitboard usOcc = pos.pieces(us), themOcc = pos.pieces(them), occ = usOcc|themOcc;
    Bitboard empty = ~occ;

    // pawns
    Bitboard pawns = pos.pieces(us, PAWN);
    int pushDir = (us==WHITE)? 8 : -8;
    int startRank = (us==WHITE)?1:6;
    int promoRank = (us==WHITE)?7:0;
    Bitboard tmp = pawns;
    while(tmp){
        Square from = pop_lsb(tmp);
        int r = rank_of(from), f = file_of(from);
        Square one = Square(from + pushDir);
        // single push
        if(one>=0&&one<64 && pos.piece_on(one)==NO_PIECE){
            if(rank_of(one)==promoRank){
                for(PieceType pt: {QUEEN,ROOK,BISHOP,KNIGHT})
                    add_move(moves, from, one, MoveFlag::PROMO, pt);
            } else {
                add_move(moves, from, one, 0);
                // double push
                if(r==startRank){
                    Square two = Square(one + pushDir);
                    if(pos.piece_on(two)==NO_PIECE) add_move(moves, from, two, MoveFlag::DOUBLE);
                }
            }
        }
        // captures
        for(int df=-1; df<=1; df+=2){
            int nf=f+df, nr=r+(us==WHITE?1:-1);
            if(nf<0||nf>=8||nr<0||nr>=8) continue;
            Square to=make_square(nf,nr);
            bool isCap = pos.piece_on(to)!=NO_PIECE && color_of(pos.piece_on(to))==them;
            bool isEP = (to==pos.ep_square());
            if(isCap || isEP){
                int fl = isEP ? MoveFlag::ENPASSANT : MoveFlag::CAPTURE;
                if(isEP) fl |= MoveFlag::CAPTURE;
                if(rank_of(to)==promoRank){
                    for(PieceType pt: {QUEEN,ROOK,BISHOP,KNIGHT})
                        add_move(moves, from, to, fl|MoveFlag::PROMO, pt);
                } else add_move(moves, from, to, fl);
            }
        }
    }

    // knights
    Bitboard knights = pos.pieces(us, KNIGHT);
    tmp = knights;
    while(tmp){
        Square from=pop_lsb(tmp);
        Bitboard att = KnightAttacks[from] & ~usOcc;
        while(att){ Square to=pop_lsb(att);
            int fl = (pos.piece_on(to)!=NO_PIECE)?MoveFlag::CAPTURE:0;
            add_move(moves, from, to, fl);
        }
    }
    // bishops
    Bitboard bishops = pos.pieces(us, BISHOP);
    tmp=bishops;
    while(tmp){
        Square from=pop_lsb(tmp);
        Bitboard att=bishop_attacks(from, occ) & ~usOcc;
        while(att){ Square to=pop_lsb(att);
            int fl=(pos.piece_on(to)!=NO_PIECE)?MoveFlag::CAPTURE:0;
            add_move(moves, from,to,fl);
        }
    }
    // rooks
    Bitboard rooks=pos.pieces(us, ROOK);
    tmp=rooks;
    while(tmp){
        Square from=pop_lsb(tmp);
        Bitboard att=rook_attacks(from, occ) & ~usOcc;
        while(att){ Square to=pop_lsb(att);
            int fl=(pos.piece_on(to)!=NO_PIECE)?MoveFlag::CAPTURE:0;
            add_move(moves, from,to,fl);
        }
    }
    // queens
    Bitboard queens=pos.pieces(us, QUEEN);
    tmp=queens;
    while(tmp){
        Square from=pop_lsb(tmp);
        Bitboard att=queen_attacks(from, occ) & ~usOcc;
        while(att){ Square to=pop_lsb(att);
            int fl=(pos.piece_on(to)!=NO_PIECE)?MoveFlag::CAPTURE:0;
            add_move(moves, from,to,fl);
        }
    }
    // king
    Square ksq = pos.king_sq(us);
    {
        Bitboard att=KingAttacks[ksq] & ~usOcc;
        while(att){ Square to=pop_lsb(att);
            // avoid moving into check is filtered in legal stage, but filter castling here
            int fl=(pos.piece_on(to)!=NO_PIECE)?MoveFlag::CAPTURE:0;
            // king cannot move adjacent to enemy king — will be filtered by legality (attack check)
            add_move(moves, ksq, to, fl);
        }
    }
    // castling (pseudo-legal, check emptiness + not in check + squares not attacked)
    if(!pos.in_check()){
        int cr = pos.castling_rights();
        if(us==WHITE){
            if(cr&1){
                // e1->g1 : f1,g1 empty, f1,g1 not attacked
                if(pos.piece_on(make_square(5,0))==NO_PIECE && pos.piece_on(make_square(6,0))==NO_PIECE){
                    if(!pos.square_attacked(make_square(5,0), them) && !pos.square_attacked(make_square(6,0), them)
                       && pos.piece_on(make_square(7,0))==W_ROOK)
                        add_move(moves, ksq, make_square(6,0), MoveFlag::CASTLING);
                }
            }
            if(cr&2){
                if(pos.piece_on(make_square(1,0))==NO_PIECE && pos.piece_on(make_square(2,0))==NO_PIECE && pos.piece_on(make_square(3,0))==NO_PIECE){
                    if(!pos.square_attacked(make_square(2,0), them) && !pos.square_attacked(make_square(3,0), them)
                       && pos.piece_on(make_square(0,0))==W_ROOK)
                        add_move(moves, ksq, make_square(2,0), MoveFlag::CASTLING);
                }
            }
        } else {
            if(cr&4){
                if(pos.piece_on(make_square(5,7))==NO_PIECE && pos.piece_on(make_square(6,7))==NO_PIECE){
                    if(!pos.square_attacked(make_square(5,7), them) && !pos.square_attacked(make_square(6,7), them)
                       && pos.piece_on(make_square(7,7))==B_ROOK)
                        add_move(moves, ksq, make_square(6,7), MoveFlag::CASTLING);
                }
            }
            if(cr&8){
                if(pos.piece_on(make_square(1,7))==NO_PIECE && pos.piece_on(make_square(2,7))==NO_PIECE && pos.piece_on(make_square(3,7))==NO_PIECE){
                    if(!pos.square_attacked(make_square(2,7), them) && !pos.square_attacked(make_square(3,7), them)
                       && pos.piece_on(make_square(0,7))==B_ROOK)
                        add_move(moves, ksq, make_square(2,7), MoveFlag::CASTLING);
                }
            }
        }
    }

    return moves;
}

std::vector<Move> generate_legal(const Position& pos){
    auto pseudo = generate_pseudo_legal(pos);
    std::vector<Move> legal; legal.reserve(pseudo.size());
    Position tmp = pos;
    for(Move m: pseudo){
        tmp.do_move(m);
        // after move, our previous king must not be in check by opponent
        Color us = pos.side_to_move();
        Square ksq = tmp.king_sq(us); // actually us king after move — need us king square from tmp but us is previous side
        // tmp side to move is them, so check if us king attacked by them
        // attackers_to with occ after move
        // find king of us in tmp
        Square myKing = tmp.king_sq(us);
        // if we moved king, myKing is new square
        bool attacked = tmp.square_attacked(myKing, tmp.side_to_move());
        if(!attacked) legal.push_back(m);
        tmp.undo_move(m);
    }
    return legal;
}

uint64_t perft(Position& pos, int depth){
    if(depth==0) return 1;
    auto moves = generate_legal(pos);
    if(depth==1) return moves.size();
    uint64_t nodes=0;
    for(Move m: moves){
        pos.do_move(m);
        nodes += perft(pos, depth-1);
        pos.undo_move(m);
    }
    return nodes;
}

Move parse_uci_move(const Position& pos, const std::string& s){
    // Strict UCI parsing: validate coordinates, promo char, and legality.
    // Returns 0 on any illegal/malformed input — caller (set_position) will emit
    // "info string illegal move" and stop applying further moves.
    if(s.size()<4 || s.size()>5) return 0;
    auto valid = [](char file, char rank){ return file>='a'&&file<='h' && rank>='1'&&rank<='8'; };
    if(!valid(s[0],s[1]) || !valid(s[2],s[3])) return 0;
    if(s.size()==5){
        char c=(char)std::tolower((unsigned char)s[4]);
        if(c!='q'&&c!='r'&&c!='b'&&c!='n') return 0;
    }
    int ff=s[0]-'a', fr=s[1]-'1', tf=s[2]-'a', tr=s[3]-'1';
    Square from=make_square(ff,fr), to=make_square(tf,tr);
    PieceType promo=QUEEN;
    bool isPromo=false;
    if(s.size()>=5){
        char c=(char)std::tolower((unsigned char)s[4]);
        if(c=='q') promo=QUEEN;
        else if(c=='r') promo=ROOK;
        else if(c=='b') promo=BISHOP;
        else if(c=='n') promo=KNIGHT;
        isPromo=true;
    }
    auto moves=generate_legal(pos);
    for(Move m: moves){
        if(move_from(m)==from && move_to(m)==to){
            if(isPromo){
                if(is_promo(m) && move_promo(m)==promo) return m;
            } else {
                if(!is_promo(m)) return m;
            }
        }
    }
    // No fallback construction: if not in legal set, it is illegal for this position.
    return 0;
}

} // namespace owen2
