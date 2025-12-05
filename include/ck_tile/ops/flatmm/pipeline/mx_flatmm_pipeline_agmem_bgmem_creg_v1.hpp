// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full,
          typename ComputeDataType_        = ADataType_>
struct MXFlatmmPipelineProblem : FlatmmPipelineProblem<ADataType_,
                                                       ADataType_,
                                                       CDataType_,
                                                       BlockGemmShape_,
                                                       Traits_,
                                                       Scheduler_,
                                                       HasHotLoop_,
                                                       TailNum_,
                                                       ComputeDataType_>
{
    using BlockGemmShape = BlockGemmShape_;

    // using QuantType = BDataType_;

    static constexpr int ScaleGranularityK = 32;

    static constexpr int ContinuousKPerThread = 32; // it's fixed for mx
    static constexpr int MXdlPack             = 2;  // it's fixed for mx
    static constexpr int NXdlPack             = 2;  // it's fixed for mx
    static constexpr int KXdlPack             = 2;
    // static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp * KXdlPack;
    static constexpr index_t flatKPerWarp = get_warp_size() * ContinuousKPerThread;
};

template <typename Problem, typename PipelinePolicy = MXFlatmmPipelineAgBgCrPolicy>
struct MXFlatmmPipelineAGmemBGmemCRegV1 : FlatmmPipelineAGmemBGmemCRegV1<Problem, PipelinePolicy>
{
    using Underlying = FlatmmPipelineAGmemBGmemCRegV1<Problem, PipelinePolicy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

    using ComputeType = ADataType;
    static_assert(sizeof(ADataType) >= sizeof(BDataType));

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    static constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

    using BlockFlatmm =
        remove_cvref_t<decltype(PipelinePolicy::template GetBlockFlatmm<Problem>())>;

    static constexpr auto config =
        BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t DsWritePreIssue = 3; // default 2, ds write at MIter - 2

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t WaveSize  = get_warp_size();

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    static constexpr index_t GetVectorSizeA() { return 32; } /* fixed for fp4 shuffle layout*/
    static constexpr index_t GetVectorSizeB() { return 32; } /* fixed for fp4 shuffle layout*/
    static constexpr index_t GetVectorSizeC() { return Problem::VectorSizeC; }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    // static constexpr index_t kLdsAlignmentInBytes = 16;
    static constexpr index_t NumWaveGroups    = Problem::NumWaveGroups;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
    static constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
    static constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

    static constexpr index_t KFlatBytesPerBlockPerIter = flatKPerWarp / BPackedSize;
    static constexpr index_t NFlatPerBlockPerIter      = flatNPerWarp;

    static constexpr index_t MPerBlockPerIter = kMPerBlock / MIterPerWarp;
    static constexpr index_t KPerBlockPerIter = kKPerBlock / KIterPerWarp;

    // static constexpr index_t WG_AKPacks = WG::kK / APackedSize;
    // static constexpr index_t WG_BKPacks = WG::kK / BPackedSize;

    static constexpr index_t MXdlPack          = Problem::MXdlPack;
    static constexpr index_t NXdlPack          = Problem::NXdlPack;
    static constexpr index_t KXdlPack          = Problem::KXdlPack;
    static constexpr index_t ScaleGranularityK = Problem::ScaleGranularityK;

    static constexpr index_t MPackIterPerWarp = MIterPerWarp / MXdlPack;
    static constexpr index_t NPackIterPerWarp = NIterPerWarp / NXdlPack;
    static constexpr index_t KPackIterPerWarp = KIterPerWarp / KXdlPack;

    static constexpr index_t AK1 = Problem::VectorLoadSize / sizeof(ADataType);
    static constexpr index_t BK1 = Problem::VectorLoadSize / sizeof(BDataType);

    static constexpr index_t a_xdl_packs     = MPackIterPerWarp * KPackIterPerWarp;
    static constexpr index_t a_preload_packs = 1;
    static constexpr index_t m_preload       = a_preload_packs * MXdlPack * KXdlPack;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;

    static constexpr index_t mfma_per_wg = 1; // 950 only

    static constexpr index_t a_dsread_per_wg = WG::kM * WG::kK / AK1 / WaveSize;
    static_assert((WG::kM * WG::kK) % (AK1 * WaveSize) == 0);
    static constexpr index_t b_dsread_per_wg = WG::kN * WG::kK / BK1 / WaveSize;
    static_assert((WG::kN * WG::kK) % (BK1 * WaveSize) == 0);

    static constexpr index_t a_dsread_num = a_dsread_per_wg * KIterPerWarp * MIterPerWarp;
    static constexpr index_t a_load_num   = a_dsread_num / NWarp;
    static constexpr index_t b_dsread_num = b_dsread_per_wg * KIterPerWarp * NIterPerWarp;
    static constexpr index_t b_load_num   = b_dsread_num / MWarp;

    static constexpr index_t ScaleBload_num =
        kNPerBlock * kKPerBlock / NWarp / ScaleGranularityK / NXdlPack / KXdlPack / WaveSize;
    static constexpr index_t ScaleAload_num =
        kMPerBlock * kKPerBlock / MWarp / ScaleGranularityK / MXdlPack / KXdlPack / WaveSize;

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

#if 0 
    CK_TILE_HOST_DEVICE static constexpr auto SchedulerPerMPack(index_t dsread_perM,
                                                                index_t load_perM)
    {
        // Init inst order
        index_t max_data_inst   = max(dsread_perM, load_perM);
        index_t sum_data_inst   = dsread_perM + load_perM;
        index_t round_data_inst = (sum_data_inst +  - 1) / ;

        index_t inst_order[NIterPerWarp * 10];
        _Pragma("unroll") for(int idx = 0; idx < NIterPerWarp * 10; idx++) { inst_order[idx] = 0; }

        index_t index = 0;
        _Pragma("unroll") for(int j = 0; j < max_data_inst; j++)
        {
            if(load_perM > j)
            {
                inst_order[index] = 2;
                index++;
            }
            if(dsread_perM > j)
            {
                inst_order[index] = 3;
                index++;
            }
        }

        // Schedule IGLP
        _Pragma("unroll") for(int j = 0; j < ; j++)
        {
            index_t inst_idx = 0;
            if(j == 0)
                ;
            else if(j == 1)
                inst_idx =  == 2 ? 1 :  - 2;
            else if(j == 2)
                inst_idx =  - 1;
            else
                inst_idx =  - j;

            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

            _Pragma("unroll") for(int r = 0; r < round_data_inst; r++)
            {
                if(r % 2 == 0)
                {
                    if(inst_order[inst_idx + r * ] == 2)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                    if(inst_order[inst_idx + r * ] == 3)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                }
                else
                {
                    if(inst_order[(r + 1) *  - 1 - inst_idx] == 2)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                    if(inst_order[(r + 1) *  - 1 - inst_idx] == 3)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                }
            }
        }
    }
#endif

    CK_TILE_HOST_DEVICE static constexpr auto HotLoopScheduler()
    {
        // Keypoint of pipeline optimize is workload balance in time
        // instruction schedule example(fp4, 128X256X256, 1X4, 16X16X128):
        // Iter MNK     MFMA  a_dsread b_dsread  A_load  b_load scale_load
        // -1   M6N2K0  56    -        -         -       -      -
        // -1   M6N3K0  57    -        -         -       -      -
        // -1   M7N2K0  58    -        -         -       -      -
        // -1   M7N3K0  59    -        -         -       -      -
        // -1   M6N2K1  60    -        -         -       -      -
        // -1   M6N3K1  61   01        -         -       -      -
        // -1   M7N2K1  62    -        -         -       -      -
        // -1   M7N3K1  63   23        -         -       -      -
        //  0   M0N0K0   0    -        4         0       -      -
        //  0   M0N1K0   1    -        5         1       -      -
        //  0   M1N0K0   2    -        -         2       -      -
        //  0   M1N1K0   3    -        -         3       -      -
        //  0   M0N0K1   4    -        6         -       4      -
        //  0   M0N1K1   5    -        7         -       5      -
        //  0   M1N0K1   6    -        -         -       6      -
        //  0   M1N1K1   7    -        -         -       7      -
        //  0   M0N2K0   8    -        -         -       8      -
        //  0   M0N3K0   9    -        -         -       -      -
        //  0   M1N2K0  10    -        -         -       9      -
        //  0   M1N3K0  11    -        -         -       -      -
        //  0   M0N2K1  12    -        -         -      10      -
        //  0   M0N3K1  13   45        -         -       -      -
        //  0   M1N2K1  14    -        -         -      11      -
        //  0   M1N3K1  15   67        -         -       -      -
        //  0   M2N0K0  16    -        -         -      12      -
        //  0   M2N1K0  17    -        -         -      13      -
        //  0   M3N0K0  18    -        -         -      14      -
        //  0   M3N1K0  19    -        -         -      15      -
        //  0   M2N0K1  20    -        -         -      16      -
        //  0   M2N1K1  21    -        -         -       -     a0
        //  0   M3N0K1  22    -        -         -       -     a1
        //  0   M3N1K1  23    -        -         -       -     a2
        //  0   M2N2K0  24    -        -         -       -     a3
        //  0   M2N3K0  25    -        -         -       -      -
        //  0   M3N2K0  26    -        -         -       -     a4
        //  0   M3N3K0  27    -        -         -       -      -
        //  0   M2N2K1  28    -        -         -       -     a5
        //  0   M2N3K1  29   89        -         -       -      -
        //  0   M3N2K1  30    -        -         -       -     a6
        //  0   M3N3K1  31 1011        -         -       -      -
        //  0   M4N0K0  32    -        -         -       -     a7
        //  0   M4N1K0  33    -        -         -       -     b0
        //  0   M5N0K0  34    -        -         -       -     b1
        //  0   M5N1K0  35    -        -         -       -     b2
        //  0   M4N0K1  36    -        -         -       -     b3
        //  0   M4N1K1  37    -        -         -       -      -
        //  0   M5N0K1  38    -        -         -       -      -
        //  0   M5N1K1  39    -        -         -       -      -
        //  0   M4N2K0  40    -        -         -       -      -
        //  0   M4N3K0  41    -        -         -       -      -
        //  0   M5N2K0  42    -        -         -       -      -
        //  0   M5N3K0  43    -        -         -       -      -
        //  0   M4N2K1  44    -        -         -       -      -
        //  0   M4N3K1  45 1213        -         -       -      -
        //  0   M5N2K1  46    -        -         -       -      -
        //  0   M5N3K1  47 1415        -         -       -      -
        //  0   M6N0K0  48    -        -         -       -      -
        //  0   M6N1K0  49    -        -         -       -      -
        //  0   M7N0K0  50    -        0         -       -      -
        //  0   M7N1K0  51    -        1         -       -      -
        //  0   M6N0K1  52    -        -         -       -      -
        //  0   M6N1K1  53    -        -         -       -      -
        //  0   M7N0K1  54    -        2         -       -      -
        //  0   M7N1K1  55    -        3         -       -      -
        //  0   M6N2K0  56    -        -         -       -      -
        //  0   M6N3K0  57    -        -         -       -      -
        //  0   M7N2K0  58    -        -         -       -      -
        //  0   M7N3K0  59    -        -         -       -      -
        //  0   M6N2K1  60    -        -         -       -      -
        //  0   M6N3K1  61 1617        -         -       -      -
        //  0   M7N2K1  62    -        -         -       -      -
        //  0   M7N3K1  63 1819        -         -       -      -

        constexpr int DSREAD_A = 0, DSREAD_B = 1, LOAD_A = 2, LOAD_B = 3, SCALE_AB = 4;
        int load_a_cnt     = a_load_num;
        int load_b_cnt     = b_load_num;
        int load_scale_cnt = ScaleAload_num + ScaleBload_num;

        constexpr int wg_cnt = KIterPerWarp * MIterPerWarp * NIterPerWarp;
        index_t queue_start  = 0;
        index_t queue_end    = 0;
        int32_t queue[wg_cnt];
        _Pragma("unroll") for(int i = 0; i < wg_cnt + 16; i++) queue[i] = 0;

        static_for_product<sequence<0, KPackIterPerWarp, 1>,
                           sequence<0, MPackIterPerWarp, 1>,
                           sequence<0, NPackIterPerWarp, 1>,
                           sequence<0, KXdlPack, 1>,
                           sequence<0, MXdlPack, 1>,
                           sequence<0, NXdlPack, 1>>{}(
            [&](auto ikpack, auto impack, auto inpack, auto ikxdl, auto imxdl, auto inxdl) {
                static_assert(mfma_per_wg == 1);
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                constexpr auto m_iter = impack * MXdlPack + imxdl;
                constexpr auto n_iter = inpack * NXdlPack + inxdl;

                if constexpr(n_iter == NIterPerWarp - 1 && ikxdl == KXdlPack - 1)
                {
                    queue[queue_end++] = DSREAD_A;
                }
                // if constexpr(m_iter == MIterPerWarp - 1 && ikxdl == KXdlPack - 1)
                // {
                //     queue[queue_end++] = DSREAD_B;
                // }

                if(queue_start == queue_end)
                {
                    if(load_a_cnt > 0)
                    {
                        queue[queue_end++] = LOAD_A;
                        load_a_cnt--;
                    }
                    else if(load_b_cnt > 0)
                    {
                        queue[queue_end++] = LOAD_B;
                        load_b_cnt--;
                    }
                    else if(load_scale_cnt > 0)
                    {
                        queue[queue_end++] = SCALE_AB;
                        load_scale_cnt--;
                    }
                }

                // consume queue
                if(queue_start < queue_end)
                {
                    int inst = queue[queue_start++];
                    if(inst == DSREAD_A || inst == DSREAD_B)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                    else if(inst == LOAD_A || inst == LOAD_B || inst == SCALE_AB)
                    {
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                }
            });
    }

#if 0 
    CK_TILE_HOST_DEVICE static constexpr auto Last2ndHotLoopScheduler()
    {
        _Pragma("unroll") for(int kIter = 0; kIter < KPackIterPerWarp; kIter++)
        {
            _Pragma("unroll") for(int mIter = 0; mIter < MIterPerWarp; mIter++)
            {
                index_t dsread_perM = 0;
                index_t load_perM   = 0;

                // Calculate ds_read number per M
                dsread_perM = a_dsread_per_wg;

                // Calculate buffer_load number per M
                if(mIter < HalfMIter)
                {
                    load_perM =
                        ((Bload_num_perK - (HalfMIter - 1 - mIter) * Bload_rep) > 0 ? Bload_rep
                                                                                    : 0);
                }
                SchedulerPerMPack(dsread_perM, load_perM);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
    }

    CK_TILE_HOST_DEVICE static constexpr auto LastHotLoopScheduler()
    {
        _Pragma("unroll") for(int kIter = 0; kIter < KIterPerWarp; kIter++)
        {
            _Pragma("unroll") for(int mIter = 0; mIter < MIterPerWarp; mIter++)
            {
                index_t dsread_perM = 0;
                index_t load_perM   = 0;

                // Calculate ds_read number per M
                if((kIter * MIterPerWarp + mIter) < (KIterPerWarp * MIterPerWarp - m_preload))
                    dsread_perM = a_dsread_per_wg;

                SchedulerPerMPack(dsread_perM, load_perM);
            }
        }
        // __builtin_amdgcn_sched_barrier(0);
    }
#endif
    CK_TILE_HOST_DEVICE static constexpr auto GetADramTileDistribution()
    {
        return PipelinePolicy::template MakeADramTileDistribution<Problem>();
    }

    template <typename... Args>
    CK_TILE_DEVICE auto operator()(void* __restrict__ smem_, Args&&... args) const
    {
        auto smem = static_cast<char*>(smem_);
        auto get_ = [&](auto size) {
            const auto prev = smem;
            smem += size;
            return prev;
        };
        auto c_warp_tensors = Run_( //
            reinterpret_cast<ADataType*>(get_(PipelinePolicy::template GetSmemSizeA<Problem>())),
            reinterpret_cast<ADataType*>(get_(PipelinePolicy::template GetSmemSizeA<Problem>())),
            reinterpret_cast<uint8_t*>(get_(PipelinePolicy::template GetSmemSizeB<Problem>())),
            reinterpret_cast<uint8_t*>(get_(PipelinePolicy::template GetSmemSizeB<Problem>())),
            std::forward<Args>(args)...);

        // Block GEMM Acc register tile
        using CWarpDstr = typename WG::CWarpDstr;
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        auto c_block_tile                   = BlockFlatmm{}.MakeCBlockTile();
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                c_block_tile.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensors(mIter)(nIter).get_thread_buffer());
            });
        });
        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp>
    CK_TILE_DEVICE auto Run_(ADataType* __restrict__ a_smem0,
                             ADataType* __restrict__ a_smem1,
                             uint8_t* __restrict__ b_smem0,
                             uint8_t* __restrict__ b_smem1,
                             const ADramBlockWindowTmp& a_copy_dram_window_tmp,
                             const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                             const ScaleADramBlockWindowTmp& scale_a_window,
                             const ScaleBDramBlockWindowTmp& scale_b_window,
                             index_t num_loop) const
    {
#ifndef __gfx950__
        static_assert(false, "Only gfx950 is supported for MXFP4 flatmm pipeline now.");
#endif
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");
        static_assert(kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // constexpr auto MIter_2nd_last = max(0, MIterPerWarp - 2);
        static_assert(NWarp == 4);

        using CWarpTensor = typename WG::CWarpTensor;

        auto a_dram_window =
            make_tile_window(PipelinePolicy::template MakeMX_AAsyncLoadDramDescriptor<Problem>(
                                 a_copy_dram_window_tmp.get_bottom_tensor_view()),
                             a_copy_dram_window_tmp.get_window_lengths(),
                             a_copy_dram_window_tmp.get_window_origin(),
                             PipelinePolicy::template MakeMX_ADramTileDistribution<Problem>());

        __builtin_amdgcn_sched_barrier(0);

        constexpr auto a_lds_block_desc =
            PipelinePolicy::template MakeMX_ALdsBlockDescriptor<Problem>();

        auto a_lds_block_ping =
            make_tensor_view<address_space_enum::lds>(a_smem0, a_lds_block_desc);
        auto a_lds_block_pong =
            make_tensor_view<address_space_enum::lds>(a_smem1, a_lds_block_desc);

        auto a_store_lds_window_ping = make_tile_window(
            a_lds_block_ping, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto a_store_lds_window_pong = make_tile_window(
            a_lds_block_pong, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // ping-pong window for A LDS
        // ping: first a_preload_packs of odd kloops + reset of even kloops
        // pong: first a_preload_packs of even kloops + reset of odd kloops
        auto a_warp_window_ping =
            make_tile_window(a_lds_block_ping,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {0, 0},
                             PipelinePolicy::template MakeMX_ALDS_TileDistribution<Problem>());
        auto a_warp_window_pong =
            make_tile_window(a_lds_block_pong,
                             make_tuple(number<WG::kM>{}, number<WG::kK>{}),
                             {0, 0},
                             PipelinePolicy::template MakeMX_ALDS_TileDistribution<Problem>());

        // B flat DRAM window for load

        // pingpong buffer for B
        auto b_flat_dram_window = PipelinePolicy::template MakeMX_BFlatBytesDramWindow<Problem>(
            b_flat_dram_block_window_tmp);

        constexpr auto b_lds_block_desc =
            PipelinePolicy::template MakeMX_BFlatBytesLdsBlockDescriptor<Problem>();
        auto b_lds_block_ping =
            make_tensor_view<address_space_enum::lds>(b_smem0, b_lds_block_desc);
        auto b_lds_block_pong =
            make_tensor_view<address_space_enum::lds>(b_smem1, b_lds_block_desc);
        auto b_store_lds_window_ping =
            make_tile_window(b_lds_block_ping, b_lds_block_desc.get_lengths(), {0, 0});
        auto b_store_lds_window_pong =
            make_tile_window(b_lds_block_pong, b_lds_block_desc.get_lengths(), {0, 0});
        // ping-pong window for B LDS
        auto b_warp_window_ping = make_tile_window(
            b_lds_block_ping,
            make_tuple(number<1>{}, number<WG::kN * WG::kK / BPackedSize>{}),
            {0, 0},
            PipelinePolicy::template MakeMX_BFlatBytesLdsTileDistribution<Problem>());
        auto b_warp_window_pong = make_tile_window(
            b_lds_block_pong,
            make_tuple(number<1>{}, number<WG::kN * WG::kK / BPackedSize>{}),
            {0, 0},
            PipelinePolicy::template MakeMX_BFlatBytesLdsTileDistribution<Problem>());

        // pingpong buffer for Scale A and Scale B
        auto scale_a_dram_window = make_tile_window(
            scale_a_window.get_bottom_tensor_view(),
            make_tuple(number<MWarp * WG::kM>{}, number<64 / WG::kM>{}),
            scale_a_window.get_window_origin(),
            PipelinePolicy::template MakeMX_ScaleA_FlatDramTileDistribution<Problem>());
        const auto scale_a_dram_step_m = amd_wave_read_first_lane(
            scale_a_dram_window.get_load_offset(tuple<number<MWarp * WG::kM>, number<0>>{}));
        const auto scale_a_dram_step_k = amd_wave_read_first_lane(
            scale_a_dram_window.get_load_offset(tuple<number<0>, number<64 / WG::kM>>{}));

        auto scale_b_dram_window = make_tile_window(
            scale_b_window.get_bottom_tensor_view(),
            make_tuple(number<NWarp * WG::kN>{}, number<64 / WG::kN>{}),
            scale_b_window.get_window_origin(),
            PipelinePolicy::template MakeMX_ScaleB_DramTileDistribution<Problem>());
        const auto scale_b_dram_step_n = amd_wave_read_first_lane(
            scale_b_dram_window.get_load_offset(tuple<number<NWarp * WG::kN>, number<0>>{}));
        const auto scale_b_dram_step_k = amd_wave_read_first_lane(
            scale_b_dram_window.get_load_offset(tuple<number<0>, number<64 / WG::kN>>{}));

        // ping pong buffer for scale A
        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(scale_a_dram_window)), KPackIterPerWarp>,
            MPackIterPerWarp>
            scale_a_tile_tensor_ping, scale_a_tile_tensor_pong;

        // ping pong buffer for scale B
        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(scale_b_dram_window)), KPackIterPerWarp>,
            NPackIterPerWarp>
            scale_b_tile_tensor_ping, scale_b_tile_tensor_pong;

        auto async_load_tile_ = [](auto lds, auto dram) {
            async_load_tile(lds, dram, number<-1>{}, true_type{}, true_type{});
        };

        // HEAD
        // Prefetch A0 first part
        static_for_product<sequence<0, KPackIterPerWarp, 1>, sequence<0, MPackIterPerWarp, 1>>{}( //
            [&](auto ikpack, auto impack) {
                constexpr auto i_pack = impack * KPackIterPerWarp + ikpack;
                if constexpr(i_pack < a_preload_packs)
                    async_load_tile_with_offset(a_store_lds_window_ping,
                                                a_dram_window,
                                                make_tuple(number<impack * MXdlPack * WG::kM>{},
                                                           number<ikpack * KXdlPack * WG::kK>{}));
            });
        // move_tile_window(a_dram_window, {0, kKPerBlock});
        async_load_tile_(b_store_lds_window_ping, b_flat_dram_window);
        move_tile_window(b_flat_dram_window, {0, KIterPerWarp * KFlatBytesPerBlockPerIter});

        // prefetch Scale A
        static_for<0, MPackIterPerWarp, 1>{}([&](auto impack) {
            static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
                scale_a_tile_tensor_ping(impack)(ikpack) = load_tile_with_offset(
                    scale_a_dram_window,
                    impack * scale_a_dram_step_m + ikpack * scale_a_dram_step_k);
            });
        });
        // move Scale A window to next K
        move_tile_window(scale_a_dram_window, {0, kKPerBlock / (32 * KXdlPack)});

        // prefetch Scale B
        static_for<0, NPackIterPerWarp, 1>{}([&](auto inpack) {
            static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
                scale_b_tile_tensor_ping(inpack)(ikpack) = load_tile_with_offset(
                    scale_b_dram_window,
                    inpack * scale_b_dram_step_n + ikpack * scale_b_dram_step_k);
            });
        });
        // move Scale B window to next K
        move_tile_window(scale_b_dram_window, {0, kKPerBlock / (32 * KXdlPack)});
        __builtin_amdgcn_sched_barrier(0);

        // Prefetch A1 B1
        constexpr bool HasSndLoop = HasHotLoop || TailNum == TailNumber::Even;
        static_for<0, a_xdl_packs, 1>{}( //
            [&](auto i_a_pack) {
                constexpr auto i_pack_with_offset = i_a_pack + a_preload_packs;
                constexpr auto next_i_pack_load   = i_pack_with_offset % a_xdl_packs;
                if(next_i_pack_load == 0 && HasSndLoop)
                    move_tile_window(a_dram_window, {0, kKPerBlock});
                constexpr auto next_impack = next_i_pack_load / KPackIterPerWarp;
                constexpr auto next_ikpack = next_i_pack_load % KPackIterPerWarp;
                constexpr auto offset      = make_tuple(number<next_impack * MXdlPack * WG::kM>{},
                                                   number<next_ikpack * KXdlPack * WG::kK>{});
                if constexpr(next_i_pack_load >= a_preload_packs || HasSndLoop)
                    // a0 next part & a1 first part
                    async_load_tile_with_offset(a_store_lds_window_pong, a_dram_window, offset);
            });
        if constexpr(HasSndLoop)
        {
            // async_load_tile_(b_store_lds_window_pong, b_flat_dram_window);
            // move_tile_window(b_flat_dram_window, {0, KIterPerWarp * KFlatBytesPerBlockPerIter});
        }
        // initialize C
        statically_indexed_array<statically_indexed_array<CWarpTensor, NIterPerWarp>, MIterPerWarp>
            c_warp_tensors;
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}(
                [&](auto nIter) { clear_tile(c_warp_tensors(mIter)(nIter)); });
        });

        statically_indexed_array<decltype(load_tile(a_warp_window_pong)), m_preload> a_warp_tensor;
        statically_indexed_array<decltype(load_tile(b_warp_window_pong)),
                                 NIterPerWarp * KIterPerWarp>
            b_warp_tensor;

        // preload A00,A10... from lds
        s_waitcnt_barrier</*vmcnt*/ 0>();
        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            constexpr auto mIter = loadIter % MXdlPack;
            constexpr auto kIter = loadIter / MXdlPack;

            a_warp_tensor(loadIter) = load_tile_with_offset(
                a_warp_window_ping, tuple<number<mIter * WG::kM>, number<kIter * WG::kK>>{});
        });

        // static_for_product<sequence<0, KPackIterPerWarp, 1>,
        //                    sequence<0, NPackIterPerWarp, 1>,
        //                    sequence<0, KXdlPack, 1>,
        //                    sequence<0, NXdlPack, 1>>{}(
        //     [&](auto ikpack, auto inpack, auto ikxdl, auto inxdl) {
        //         constexpr auto nIter     = inpack * NXdlPack + inxdl;
        //         constexpr auto kIter     = ikpack * KXdlPack + ikxdl;
        //         constexpr auto BwarpIter = kIter * NIterPerWarp + nIter;
        //         constexpr auto nOffset   = inpack * NXdlPack * NWarp + inxdl;

        //         b_warp_tensor(number<BwarpIter>{}) = load_tile_with_offset(
        //             b_warp_window_ping,
        //             tuple<number<nOffset>, number<kIter * WG::kN * WG::kK / BPackedSize>>{});
        //     });

        __builtin_amdgcn_sched_barrier(0);

        // MAIN LOOP
        auto main_body_implx2 = [&]() mutable {
            __builtin_amdgcn_sched_barrier(0);
            // prefetch Scale A and Scale B (2i+1)
            static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
                static_for<0, MPackIterPerWarp, 1>{}([&](auto impack) {
                    scale_a_tile_tensor_pong(impack)(ikpack) = load_tile_with_offset(
                        scale_a_dram_window,
                        impack * scale_a_dram_step_m + ikpack * scale_a_dram_step_k);
                });
            });
            static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
                static_for<0, NPackIterPerWarp, 1>{}([&](auto inpack) {
                    scale_b_tile_tensor_pong(inpack)(ikpack) = load_tile_with_offset(
                        scale_b_dram_window,
                        inpack * scale_b_dram_step_n + ikpack * scale_b_dram_step_k);
                });
            });

            // Prefetch B(2i+1)
            async_load_tile_(b_store_lds_window_pong, b_flat_dram_window);
            move_tile_window(b_flat_dram_window, {0, KIterPerWarp * KFlatBytesPerBlockPerIter});

            // async load A(2i+1) next part & A(2i+2) first part
            static_for<0, a_xdl_packs, 1>{}([&](auto i_a_pack) {
                constexpr auto i_pack_with_offset = i_a_pack + a_preload_packs;
                constexpr auto next_i_pack_load   = i_pack_with_offset % a_xdl_packs;
                if(next_i_pack_load == 0)
                    move_tile_window(a_dram_window, {0, kKPerBlock});
                constexpr auto next_impack = next_i_pack_load / KPackIterPerWarp;
                constexpr auto next_ikpack = next_i_pack_load % KPackIterPerWarp;
                constexpr auto offset      = make_tuple(number<next_impack * MXdlPack * WG::kM>{},
                                                   number<next_ikpack * KXdlPack * WG::kK>{});
                async_load_tile_with_offset(a_store_lds_window_ping, a_dram_window, offset);
            });

            __builtin_amdgcn_sched_barrier(0);
            // preload B(2i) from lds
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto inpack, auto ikxdl, auto inxdl) {
                    constexpr auto nIter     = inpack * NXdlPack + inxdl;
                    constexpr auto kIter     = ikpack * KXdlPack + ikxdl;
                    constexpr auto BwarpIter = kIter * NIterPerWarp + nIter;
                    constexpr auto nOffset   = inpack * NXdlPack * NWarp + inxdl;

                    b_warp_tensor(number<BwarpIter>{}) = load_tile_with_offset(
                        b_warp_window_ping,
                        tuple<number<nOffset>, number<kIter * WG::kN * WG::kK / BPackedSize>>{});
                });

            // GEMM 2i
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, MPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, MXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto impack, auto inpack, auto ikxdl, auto imxdl, auto inxdl) {
                    constexpr auto n_iter    = inpack * NXdlPack + inxdl;
                    constexpr auto m_iter    = impack * MXdlPack + imxdl;
                    constexpr auto k_iter    = ikpack * KXdlPack + ikxdl;
                    constexpr auto APackIter = ikxdl * MXdlPack + imxdl; // idx inside a xdl pack
                    constexpr auto BwarpIter = k_iter * NIterPerWarp + n_iter;
                    //  warp GEMM
                    WG{}.template operator()<APackIter, ikxdl * NXdlPack + inxdl>(
                        c_warp_tensors(number<m_iter>{})(number<n_iter>{}),
                        bit_cast<typename WG::AWarpTensor>(a_warp_tensor(number<APackIter>{})),
                        bit_cast<typename WG::BWarpTensor>(b_warp_tensor(number<BwarpIter>{})),
                        scale_a_tile_tensor_ping(impack)(ikpack).get_thread_buffer()[0],
                        scale_b_tile_tensor_ping(inpack)(ikpack).get_thread_buffer()[0]);
                    { // a dram and lds load
                        constexpr auto i_pack_with_offset =
                            impack * KPackIterPerWarp + ikpack + a_preload_packs;
                        constexpr auto next_i_pack_load = i_pack_with_offset % a_xdl_packs;
                        constexpr auto next_impack      = next_i_pack_load / KPackIterPerWarp;
                        constexpr auto next_ikpack      = next_i_pack_load % KPackIterPerWarp;
                        if constexpr(n_iter == NIterPerWarp - 1)
                        { // lds load next A(2i) next part & A(2i+1) first part
                            constexpr auto offset =
                                make_tuple(number<(next_impack * MXdlPack + imxdl) * WG::kM>{},
                                           number<(next_ikpack * KXdlPack + ikxdl) * WG::kK>{});
                            a_warp_tensor(number<APackIter>{}) =
                                load_tile_with_offset(a_warp_window_pong, offset);
                        }
                    }
                });
            // move ab scale window to next K
            move_tile_window(scale_a_dram_window, {0, kKPerBlock / (32 * KXdlPack)});
            move_tile_window(scale_b_dram_window, {0, kKPerBlock / (32 * KXdlPack)});
            // barrier as ds_load A(2i) and buffer_load_lds A(2i + 1) finished
            s_waitcnt</*vmcnt*/ ScaleAload_num + ScaleBload_num>();
            block_sync_lds();
            HotLoopScheduler();
            __builtin_amdgcn_sched_barrier(0);

            ////////////////////////////// Next K //////////////////////////////

            // Prefetch B(2i+2)
            async_load_tile_(b_store_lds_window_ping, b_flat_dram_window);
            move_tile_window(b_flat_dram_window, {0, KIterPerWarp * KFlatBytesPerBlockPerIter});

            // preload B(2i+1) from lds
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto inpack, auto ikxdl, auto inxdl) {
                    constexpr auto nIter     = inpack * NXdlPack + inxdl;
                    constexpr auto kIter     = ikpack * KXdlPack + ikxdl;
                    constexpr auto BwarpIter = kIter * NIterPerWarp + nIter;
                    constexpr auto nOffset   = inpack * NXdlPack * NWarp + inxdl;

                    b_warp_tensor(number<BwarpIter>{}) = load_tile_with_offset(
                        b_warp_window_pong,
                        tuple<number<nOffset>, number<kIter * WG::kN * WG::kK / BPackedSize>>{});
                });
            static_for<0, a_xdl_packs, 1>{}( // async load A(2i+2) next part & A(2i+3) first part
                [&](auto i_a_pack) {
                    constexpr auto i_pack_with_offset = i_a_pack + a_preload_packs;
                    constexpr auto next_i_pack_load   = i_pack_with_offset % a_xdl_packs;
                    if(next_i_pack_load == 0)
                        move_tile_window(a_dram_window, {0, kKPerBlock});
                    constexpr auto next_impack = next_i_pack_load / KPackIterPerWarp;
                    constexpr auto next_ikpack = next_i_pack_load % KPackIterPerWarp;
                    constexpr auto offset = make_tuple(number<next_impack * MXdlPack * WG::kM>{},
                                                       number<next_ikpack * KXdlPack * WG::kK>{});
                    async_load_tile_with_offset(a_store_lds_window_pong, a_dram_window, offset);
                });

            // prefetch Scale A and Scale B (2i+2)
            static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
                static_for<0, MPackIterPerWarp, 1>{}([&](auto impack) {
                    scale_a_tile_tensor_ping(impack)(ikpack) = load_tile_with_offset(
                        scale_a_dram_window,
                        impack * scale_a_dram_step_m + ikpack * scale_a_dram_step_k);
                });
            });

            static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
                static_for<0, NPackIterPerWarp, 1>{}([&](auto inpack) {
                    scale_b_tile_tensor_ping(inpack)(ikpack) = load_tile_with_offset(
                        scale_b_dram_window,
                        inpack * scale_b_dram_step_n + ikpack * scale_b_dram_step_k);
                });
            });

            // GEMM 2i+1
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, MPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, MXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto impack, auto inpack, auto ikxdl, auto imxdl, auto inxdl) {
                    constexpr auto m_iter    = impack * MXdlPack + imxdl;
                    constexpr auto n_iter    = inpack * NXdlPack + inxdl;
                    constexpr auto k_iter    = ikpack * KXdlPack + ikxdl;
                    constexpr auto APackIter = ikxdl * MXdlPack + imxdl; // idx inside a xdl pack
                    constexpr auto BwarpIter = k_iter * NIterPerWarp + n_iter;
                    // warp GEMM
                    WG{}.template operator()<APackIter, ikxdl * NXdlPack + inxdl>(
                        c_warp_tensors(number<m_iter>{})(number<n_iter>{}),
                        bit_cast<typename WG::AWarpTensor>(a_warp_tensor(number<APackIter>{})),
                        bit_cast<typename WG::BWarpTensor>(b_warp_tensor(number<BwarpIter>{})),
                        scale_a_tile_tensor_pong(impack)(ikpack).get_thread_buffer()[0],  // scale A
                        scale_b_tile_tensor_pong(inpack)(ikpack).get_thread_buffer()[0]); // scale B
                    { // a dram and lds load
                        constexpr auto i_pack_with_offset =
                            impack * KPackIterPerWarp + ikpack + a_preload_packs;
                        constexpr auto next_i_pack_load = i_pack_with_offset % a_xdl_packs;
                        constexpr auto next_impack      = next_i_pack_load / KPackIterPerWarp;
                        constexpr auto next_ikpack      = next_i_pack_load % KPackIterPerWarp;
                        if constexpr(n_iter == NIterPerWarp - 1)
                        { // lds load next A(2i+1) next part & A(2i+2) first part
                            constexpr auto offset =
                                make_tuple(number<(next_impack * MXdlPack + imxdl) * WG::kM>{},
                                           number<(next_ikpack * KXdlPack + ikxdl) * WG::kK>{});
                            a_warp_tensor(number<APackIter>{}) =
                                load_tile_with_offset(a_warp_window_ping, offset);
                        }
                    }
                });
            // move ab scale window to next K
            move_tile_window(scale_a_dram_window, {0, kKPerBlock / (32 * KXdlPack)});
            move_tile_window(scale_b_dram_window, {0, kKPerBlock / (32 * KXdlPack)});
            // barrier as ds_load A(2i + 1) and buffer_load_lds A(2i + 2) finished
            s_waitcnt</*vmcnt*/ ScaleAload_num + ScaleBload_num>();
            block_sync_lds();
            HotLoopScheduler();
            __builtin_amdgcn_sched_barrier(0);
        };

        if constexpr(HasHotLoop)
        {
            index_t iCounter = (num_loop - 1) / 2;
            do
            {
                main_body_implx2();
                iCounter--;
            } while(iCounter > 0);
        }

        // TAIL
        if constexpr(TailNum == TailNumber::Even)
        {
            // Prefetch B(2i+1)
            async_load_tile_(b_store_lds_window_pong, b_flat_dram_window);
            move_tile_window(b_flat_dram_window, {0, KIterPerWarp * KFlatBytesPerBlockPerIter});
            // preload B(2i) from lds
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto inpack, auto ikxdl, auto inxdl) {
                    constexpr auto nIter     = inpack * NXdlPack + inxdl;
                    constexpr auto kIter     = ikpack * KXdlPack + ikxdl;
                    constexpr auto BwarpIter = kIter * NIterPerWarp + nIter;
                    constexpr auto nOffset   = inpack * NXdlPack * NWarp + inxdl;

                    b_warp_tensor(number<BwarpIter>{}) = load_tile_with_offset(
                        b_warp_window_ping,
                        tuple<number<nOffset>, number<kIter * WG::kN * WG::kK / BPackedSize>>{});
                });

            static_for<0, a_xdl_packs, 1>{}( // async load A(2i+1) next part & A(2i+2) first part
                [&](auto i_a_pack) {         // TODO(Yi): remove unused load
                    constexpr auto i_pack_with_offset = i_a_pack + a_preload_packs;
                    constexpr auto next_i_pack_load   = i_pack_with_offset % a_xdl_packs;
                    if(next_i_pack_load == 0)
                        move_tile_window(a_dram_window, {0, kKPerBlock});
                    constexpr auto next_impack = next_i_pack_load / KPackIterPerWarp;
                    constexpr auto next_ikpack = next_i_pack_load % KPackIterPerWarp;
                    constexpr auto offset = make_tuple(number<next_impack * MXdlPack * WG::kM>{},
                                                       number<next_ikpack * KXdlPack * WG::kK>{});
                    async_load_tile_with_offset(a_store_lds_window_ping, a_dram_window, offset);
                });

            // prefetch Scale A and Scale B (2i+1)
            static_for<0, MPackIterPerWarp, 1>{}([&](auto impack) {
                static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
                    scale_a_tile_tensor_pong(impack)(ikpack) = load_tile_with_offset(
                        scale_a_dram_window,
                        impack * scale_a_dram_step_m + ikpack * scale_a_dram_step_k);
                });
            });
            static_for<0, NPackIterPerWarp, 1>{}([&](auto inpack) {
                static_for<0, KPackIterPerWarp, 1>{}([&](auto ikpack) {
                    scale_b_tile_tensor_pong(inpack)(ikpack) = load_tile_with_offset(
                        scale_b_dram_window,
                        inpack * scale_b_dram_step_n + ikpack * scale_b_dram_step_k);
                });
            });

            // GEMM loopK-1
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, MPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, MXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto impack, auto inpack, auto ikxdl, auto imxdl, auto inxdl) {
                    constexpr auto m_iter    = impack * MXdlPack + imxdl;
                    constexpr auto n_iter    = inpack * NXdlPack + inxdl;
                    constexpr auto k_iter    = ikpack * KXdlPack + ikxdl;
                    constexpr auto APackIter = ikxdl * MXdlPack + imxdl; // idx inside a xdl pack
                    constexpr auto BwarpIter = k_iter * NIterPerWarp + n_iter;
                    // warp GEMM
                    WG{}.template operator()<APackIter, ikxdl * NXdlPack + inxdl>(
                        c_warp_tensors(number<m_iter>{})(number<n_iter>{}),
                        bit_cast<typename WG::AWarpTensor>(a_warp_tensor(number<APackIter>{})),
                        bit_cast<typename WG::BWarpTensor>(b_warp_tensor(number<BwarpIter>{})),
                        scale_a_tile_tensor_ping(impack)(ikpack).get_thread_buffer()[0],  // scale A
                        scale_b_tile_tensor_ping(inpack)(ikpack).get_thread_buffer()[0]); // scale B
                    { // a dram and lds load
                        constexpr auto i_pack_with_offset =
                            impack * KPackIterPerWarp + ikpack + a_preload_packs;
                        constexpr auto next_i_pack_load = i_pack_with_offset % a_xdl_packs;
                        constexpr auto next_impack      = next_i_pack_load / KPackIterPerWarp;
                        constexpr auto next_ikpack      = next_i_pack_load % KPackIterPerWarp;
                        if constexpr(n_iter == NIterPerWarp - 1)
                        { // lds load next A(2i) next part & A(2i+1) first part
                            constexpr auto offset =
                                make_tuple(number<(next_impack * MXdlPack + imxdl) * WG::kM>{},
                                           number<(next_ikpack * KXdlPack + ikxdl) * WG::kK>{});
                            a_warp_tensor(number<APackIter>{}) =
                                load_tile_with_offset(a_warp_window_pong, offset);
                        }
                    }
                });
            // barrier as ds_load A(2i) and buffer_load_lds A(2i + 1) finished
            s_waitcnt</*vmcnt*/ ScaleAload_num + ScaleBload_num>();
            block_sync_lds();
            // Last2ndHotLoopScheduler();

            // preload A(2i+1) and B(2i+1) from lds
            static_for<0, m_preload, 1>{}([&](auto loadIter) {
                constexpr auto mIter    = loadIter % MXdlPack;
                constexpr auto kIter    = loadIter / MXdlPack;
                a_warp_tensor(loadIter) = load_tile_with_offset(
                    a_warp_window_pong, tuple<number<mIter * WG::kM>, number<kIter * WG::kK>>{});
            });
            // preload B(2i+1) from lds
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto inpack, auto ikxdl, auto inxdl) {
                    constexpr auto nIter     = inpack * NXdlPack + inxdl;
                    constexpr auto kIter     = ikpack * KXdlPack + ikxdl;
                    constexpr auto BwarpIter = kIter * NIterPerWarp + nIter;
                    constexpr auto nOffset   = inpack * NXdlPack * NWarp + inxdl;

                    b_warp_tensor(number<BwarpIter>{}) = load_tile_with_offset(
                        b_warp_window_pong,
                        tuple<number<nOffset>, number<kIter * WG::kN * WG::kK / BPackedSize>>{});
                });

            // GEMM loopK
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, MPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, MXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto impack, auto inpack, auto ikxdl, auto imxdl, auto inxdl) {
                    constexpr auto m_iter    = impack * MXdlPack + imxdl;
                    constexpr auto n_iter    = inpack * NXdlPack + inxdl;
                    constexpr auto k_iter    = ikpack * KXdlPack + ikxdl;
                    constexpr auto APackIter = ikxdl * MXdlPack + imxdl; // idx inside a xdl pack
                    constexpr auto BwarpIter = k_iter * NIterPerWarp + n_iter;
                    // warp GEMM
                    WG{}.template operator()<APackIter, ikxdl * NXdlPack + inxdl>(
                        c_warp_tensors(number<m_iter>{})(number<n_iter>{}),
                        bit_cast<typename WG::AWarpTensor>(a_warp_tensor(number<APackIter>{})),
                        bit_cast<typename WG::BWarpTensor>(b_warp_tensor(number<BwarpIter>{})),
                        scale_a_tile_tensor_pong(impack)(ikpack).get_thread_buffer()[0],  // scale A
                        scale_b_tile_tensor_pong(inpack)(ikpack).get_thread_buffer()[0]); // scale B
                    { // a dram and lds load
                        constexpr auto i_pack_with_offset =
                            impack * KPackIterPerWarp + ikpack + a_preload_packs;
                        constexpr auto next_i_pack_load = i_pack_with_offset % a_xdl_packs;
                        constexpr auto next_impack      = next_i_pack_load / KPackIterPerWarp;
                        constexpr auto next_ikpack      = next_i_pack_load % KPackIterPerWarp;
                        if constexpr(n_iter == NIterPerWarp - 1) // TODO(Yi): remove unused load
                        { // lds load next A(2i) next part & A(2i+1) first part
                            constexpr auto offset =
                                make_tuple(number<(next_impack * MXdlPack + imxdl) * WG::kM>{},
                                           number<(next_ikpack * KXdlPack + ikxdl) * WG::kK>{});
                            a_warp_tensor(number<APackIter>{}) =
                                load_tile_with_offset(a_warp_window_ping, offset);
                        }
                    }
                });
            // LastHotLoopScheduler();
        }
        else if constexpr(TailNum == TailNumber::Odd)
        {
            // if(get_thread_id() < 64)
            //     printf("Odd Tail\n");
            // GEMM loopK
            // preload B(2i) from lds
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto inpack, auto ikxdl, auto inxdl) {
                    constexpr auto nIter     = inpack * NXdlPack + inxdl;
                    constexpr auto kIter     = ikpack * KXdlPack + ikxdl;
                    constexpr auto BwarpIter = kIter * NIterPerWarp + nIter;
                    constexpr auto nOffset   = inpack * NXdlPack * NWarp + inxdl;

                    b_warp_tensor(number<BwarpIter>{}) = load_tile_with_offset(
                        b_warp_window_ping,
                        tuple<number<nOffset>, number<kIter * WG::kN * WG::kK / BPackedSize>>{});
                });
            static_for_product<sequence<0, KPackIterPerWarp, 1>,
                               sequence<0, MPackIterPerWarp, 1>,
                               sequence<0, NPackIterPerWarp, 1>,
                               sequence<0, KXdlPack, 1>,
                               sequence<0, MXdlPack, 1>,
                               sequence<0, NXdlPack, 1>>{}(
                [&](auto ikpack, auto impack, auto inpack, auto ikxdl, auto imxdl, auto inxdl) {
                    constexpr auto m_iter    = impack * MXdlPack + imxdl;
                    constexpr auto n_iter    = inpack * NXdlPack + inxdl;
                    constexpr auto k_iter    = ikpack * KXdlPack + ikxdl;
                    constexpr auto APackIter = ikxdl * MXdlPack + imxdl; // idx inside a xdl pack
                    constexpr auto BwarpIter = k_iter * NIterPerWarp + n_iter;
                    // warp GEMM
                    WG{}.template operator()<APackIter, ikxdl * NXdlPack + inxdl>(
                        c_warp_tensors(number<m_iter>{})(number<n_iter>{}),
                        bit_cast<typename WG::AWarpTensor>(a_warp_tensor(number<APackIter>{})),
                        bit_cast<typename WG::BWarpTensor>(b_warp_tensor(number<BwarpIter>{})),
                        scale_a_tile_tensor_ping(impack)(ikpack).get_thread_buffer()[0],  // scale A
                        scale_b_tile_tensor_ping(inpack)(ikpack).get_thread_buffer()[0]); // scale B
                    { // a dram and lds load
                        constexpr auto i_pack_with_offset =
                            impack * KPackIterPerWarp + ikpack + a_preload_packs;
                        constexpr auto next_i_pack_load = i_pack_with_offset % a_xdl_packs;
                        constexpr auto next_impack      = next_i_pack_load / KPackIterPerWarp;
                        constexpr auto next_ikpack      = next_i_pack_load % KPackIterPerWarp;
                        if constexpr(n_iter == NIterPerWarp - 1) // TODO(Yi): remove unused load
                        { // lds load next A(2i) next part & A(2i+1) first part
                            constexpr auto offset =
                                make_tuple(number<(next_impack * MXdlPack + imxdl) * WG::kM>{},
                                           number<(next_ikpack * KXdlPack + ikxdl) * WG::kK>{});
                            a_warp_tensor(number<APackIter>{}) =
                                load_tile_with_offset(a_warp_window_pong, offset);
                            // if(get_thread_id() < 64)
                            //     CK_PRINTF<>{}(a_warp_tensor(number<APackIter>{}));
                        }
                    }
                });
            // LastHotLoopScheduler();
        }
        else
        {
            static_assert(false, "Wrong TailNum");
        }
        return c_warp_tensors;
    }
};

} // namespace ck_tile
