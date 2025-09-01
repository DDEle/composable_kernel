// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

struct GemmMXConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 2;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC = false;
    static constexpr int kBlockPerCu = 1;
};

struct ExampleArgs
{
    int M, N, K;
    int validate;
    int n_warmup;
    int n_repeat;
    int init_method;

    ExampleArgs(int argc, char* argv[])
    {
        ck_tile::ArgParser arg_parser;
        arg_parser.insert("m", "256", "m dimension")
            .insert("n", "256", "n dimension")
            .insert("k", "256", "k dimension")
            .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
            .insert("warmup", "50", "number of iterations before benchmark the kernel")
            .insert("repeat", "100", "number of iterations to benchmark the kernel")
            .insert("init", "0", "0:random, 1:constant(1)");
        if(!arg_parser.parse(argc, argv))
            throw std::runtime_error("Error: failed to parse input args");

        M           = std::stoi(arg_parser.retrieve("m"));
        N           = std::stoi(arg_parser.retrieve("n"));
        K           = std::stoi(arg_parser.retrieve("k"));
        validate    = std::stoi(arg_parser.retrieve("v"));
        n_warmup    = std::stoi(arg_parser.retrieve("warmup"));
        n_repeat    = std::stoi(arg_parser.retrieve("repeat"));
        init_method = std::stoi(arg_parser.retrieve("init"));
    }
}
