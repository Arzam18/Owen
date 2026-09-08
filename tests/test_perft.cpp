#include "../src/position.h"
#include "../src/movegen.h"
#include "../src/bitboard.h"
#include <cassert>
#include <iostream>
using namespace owen2;
int main(){
    init_attacks(); Position::init_zobrist();
    Position p;
    // Kiwipete
    p.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    Position t=p;
    uint64_t n=perft(t,3);
    std::cout<<"kiwipete perft 3 = "<<n<<" expect 97862\n";
    assert(n==97862);
    std::cout<<"perft tests passed\n";
    return 0;
}
