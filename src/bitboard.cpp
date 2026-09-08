#include "bitboard.h"

namespace owen2 {

std::array<Bitboard,64> KnightAttacks{};
std::array<Bitboard,64> KingAttacks{};
std::array<Bitboard,64> PawnAttacksWhite{};
std::array<Bitboard,64> PawnAttacksBlack{};

static Bitboard sliding_attacks(Square sq, Bitboard occ, const int dirs[4][2], int ndirs){
    Bitboard attacks=0;
    int r0=rank_of(sq), f0=file_of(sq);
    for(int d=0;d<ndirs;++d){
        int r=r0+dirs[d][0], f=f0+dirs[d][1];
        while(r>=0&&r<8&&f>=0&&f<8){
            Square s=make_square(f,r);
            attacks |= sq_bb(s);
            if(occ & sq_bb(s)) break;
            r+=dirs[d][0]; f+=dirs[d][1];
        }
    }
    return attacks;
}

Bitboard bishop_attacks(Square sq, Bitboard occ){
    const int dirs[4][2]={{1,1},{1,-1},{-1,1},{-1,-1}};
    return sliding_attacks(sq,occ,dirs,4);
}
Bitboard rook_attacks(Square sq, Bitboard occ){
    const int dirs[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    return sliding_attacks(sq,occ,dirs,4);
}

void init_attacks(){
    for(int sq=0;sq<64;++sq){
        int r=rank_of(sq), f=file_of(sq);
        Bitboard k=0,n=0,pw=0,pb=0;
        // king
        for(int dr=-1;dr<=1;++dr) for(int df=-1;df<=1;++df){
            if(dr==0&&df==0) continue;
            int nr=r+dr,nf=f+df;
            if(nr>=0&&nr<8&&nf>=0&&nf<8) k |= sq_bb(make_square(nf,nr));
        }
        // knight
        const int kd[8][2]={{2,1},{2,-1},{-2,1},{-2,-1},{1,2},{1,-2},{-1,2},{-1,-2}};
        for(auto &d: kd){
            int nr=r+d[0], nf=f+d[1];
            if(nr>=0&&nr<8&&nf>=0&&nf<8) n |= sq_bb(make_square(nf,nr));
        }
        // pawns (attacks FROM square: white pawns attack north)
        if(r<7){
            if(f>0) pw |= sq_bb(make_square(f-1,r+1));
            if(f<7) pw |= sq_bb(make_square(f+1,r+1));
        }
        if(r>0){
            if(f>0) pb |= sq_bb(make_square(f-1,r-1));
            if(f<7) pb |= sq_bb(make_square(f+1,r-1));
        }
        KingAttacks[sq]=k;
        KnightAttacks[sq]=n;
        PawnAttacksWhite[sq]=pw;
        PawnAttacksBlack[sq]=pb;
    }
}

} // namespace owen2
