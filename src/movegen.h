#pragma once
#include "position.h"
#include <vector>

namespace owen2 {

std::vector<Move> generate_legal(const Position& pos);
std::vector<Move> generate_pseudo_legal(const Position& pos);
uint64_t perft(Position& pos, int depth);
Move parse_uci_move(const Position& pos, const std::string& s);

} // namespace owen2
