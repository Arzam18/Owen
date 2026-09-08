#pragma once
#include "features.h"
#include "../position.h"
#include <string>
#include <vector>
#include <cstdint>

namespace owen2::nnue {

// Owen2 v2 — SFNNv10-class .o2nn
// magic "O2NN" ver=2, H=1024
// Layout: feature_weights [INPUT_SIZE*H] int16
//         feature_bias    [H] int16
//         l1 [H*L1] int8, L1b [L1] int16, l2 [L1*L2] int8, L2b [L2] int16,
//         out [L2] int8, out_bias int16
// v1 (256) cannot be loaded as v2 — trainer will convert if needed.

struct Network {
    static constexpr int H = HIDDEN_SIZE; // 1024
    static constexpr int L1 = 16, L2 = 32; // 1024->16->32->1 : SFNNv10 style bottleneck, faster on AVX2
    std::vector<int16_t> feature_weights; // INPUT*H
    std::array<int16_t, H> feature_bias{};
    std::array<int8_t, H*L1> l1_weights{};
    std::array<int16_t, L1> l1_bias{};
    std::array<int8_t, L1*L2> l2_weights{};
    std::array<int16_t, L2> l2_bias{};
    std::array<int8_t, L2> out_weights{};
    int16_t out_bias=0;
    bool loaded=false;
    bool load(const std::string& path);
    bool load_from_memory(const unsigned char* data, size_t size);
    bool save(const std::string& path) const;
    int evaluate(const Position& pos) const;
    int evaluate(const Position& pos, Accumulator& acc) const;
    int evaluate_handcrafted(const Position& pos) const;
private:
    int forward(const std::array<int16_t,H>& acc) const;
};

extern Network g_network;

} // namespace owen2::nnue
