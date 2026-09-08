#include "features.h"

namespace owen2::nnue {

int feature_index(Color perspective, Square kingSq, Piece piece, Square pieceSq){
    auto orient = [&](Square s)->Square { return perspective==BLACK ? Square(s ^ 56) : s; };
    Square k = orient(kingSq);
    Square ps = orient(pieceSq);
    int pc = (int)piece;
    if(pc==W_KING || pc==B_KING) return -1;
    int pc10 = pc < 6 ? pc : (pc-6)+5;
    if(pc10<0||pc10>=10) return -1;
    return pc10*4096 + k*64 + ps; // 0..40959
}
int threat_index(Color perspective, Square kingSq, Piece piece, Square pieceSq){
    int base = feature_index(perspective, kingSq, piece, pieceSq);
    if(base < 0) return -1;
    return HALFKP_SIZE + base; // 40960..81919
}

void refresh_accumulator(const Position& pos, Accumulator& acc, const int16_t* weights){
    acc.white.fill(0); acc.black.fill(0);
    Square wk = pos.king_sq(WHITE), bk = pos.king_sq(BLACK);
    for(int s=0;s<64;++s){
        Piece p = pos.piece_on(Square(s));
        if(p==NO_PIECE) continue;
        if(type_of(p)==KING) continue;
        int idxW = feature_index(WHITE, wk, p, Square(s));
        int idxB = feature_index(BLACK, bk, p, Square(s));
        if(idxW>=0){
            const int16_t* w = weights + idxW * HIDDEN_SIZE;
            for(int i=0;i<HIDDEN_SIZE;++i) acc.white[i] += w[i];
        }
        if(idxB>=0){
            const int16_t* w = weights + idxB * HIDDEN_SIZE;
            for(int i=0;i<HIDDEN_SIZE;++i) acc.black[i] += w[i];
        }
        // Threat inputs: + second plane if piece is threatened
        if(is_threatened(pos, Square(s), WHITE) && idxW>=0){
            int tW = threat_index(WHITE, wk, p, Square(s));
            const int16_t* w = weights + tW * HIDDEN_SIZE;
            for(int i=0;i<HIDDEN_SIZE;++i) acc.white[i] += w[i];
        }
        if(is_threatened(pos, Square(s), BLACK) && idxB>=0){
            int tB = threat_index(BLACK, bk, p, Square(s));
            const int16_t* w = weights + tB * HIDDEN_SIZE;
            for(int i=0;i<HIDDEN_SIZE;++i) acc.black[i] += w[i];
        }
    }
    acc.computed_white=acc.computed_black=true;
}

} // namespace owen2::nnue
