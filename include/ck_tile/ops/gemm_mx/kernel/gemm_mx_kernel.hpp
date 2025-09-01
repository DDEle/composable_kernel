// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

struct GemmMxHostArgs
{
    const void *a_ptr, b_ptr, a_scale_ptr, b_scale_ptr;
    void* c_ptr;

    index_t M, N, K;
    index_t stride_A, stride_B, stride_C;
};

using GemmMxKernelArgs = GemmMxHostArgs;

template <typename TilePartitioner_, typename GemmMXPipeline_, typename EpiloguePipeline_>
struct GemmMxKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmMxPipeline   = remove_cvref_t<GemmMXPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using BlockGemmShape   = typename GemmMxPipeline::BlockGemmShape; // TileFlatmmShape
    using ALayout          = typename GemmMxPipeline::ALayout;
    using BLayout          = typename GemmMxPipeline::BLayout;
    using ELayout          = typename GemmMxPipeline::CLayout;

    using ADataType = typename GemmMxPipeline::ADataType;
    using BDataType = typename GemmMxPipeline::BDataType;
    using EDataType = typename EpiloguePipeline::ODataType;
    

    static constexpr index_t kBlockSize = GemmMxPipeline::BlockSize;
    static constexpr index_t MPerBlock = TilePartitioner::MPerBlock;
    static constexpr index_t NPerBlock = TilePartitioner::NPerBlock;
    static constexpr index_t KPerBlock = TilePartitioner::KPerBlock;
    using KernelArgs = GemmMxKernelArgs;

    using DsLayout         = typename EpiloguePipeline::DsLayout;
    using DsDataType       = typename EpiloguePipeline::DsDataType;
    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");
    static_assert(DsLayout::size() == 0, "Support D tensor later");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_mx", gemm_prec_str<ADataType, BDataType>, GemmMxPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static constexpr KernelArgs MakeKernelArgs(const GemmMxHostArgs& hostArgs)
    {
        return hostArgs;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmMxPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs) { return true; }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const ADataType* a_ptr,
                        const BDataType* b_flat_ptr,
                        const std::array<const void*, NumDTensor>& ds_ptr,
                        EDataType* e_ptr,
                        const KernelArgs& kargs,
                        const SplitKBatchOffset& splitk_batch_offset)
    {
        // TODO: enable vector write for C in ColMajor
        const auto& e_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_E, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    e_ptr,
                    make_tuple(kargs.N, kargs.M),
                    make_tuple(kargs.stride_E, 1),
                    number<1>{},
                    number<1>{});
            }
        }();

        return make_tuple(a_tensor_view, b_flat_tensor_view, ds_tensor_view, e_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        // TODO vector write in for C in ColMajor
        const auto& e_pad_view = [&]() {
            const auto& e_tensor_view = views.at(I3);
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                       sequence<false, GemmMxPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                       sequence<GemmMxPipeline::kPadM, false>{});
            }
        }();

        return make_tuple(a_pad_view, b_flat_tensor_view, ds_pad_view, e_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        auto e_block_window = make_tile_window(
            e_pad_view, make_tuple(number<MPerBlock>{}, number<NPerBlock>{}), {i_m, i_n});

        return make_tuple(a_block_window, b_flat_block_window, ds_block_window, e_block_window);
    }

    CK_TILE_DEVICE static void
    RunGemmMx(void* smem_ptr, const KernelArgs& kargs, const index_t i_m, const index_t i_n)
    {
        const auto a_window = [&]() {
            constexpr bool is_a_row_major = std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>;
            const auto naive_view         = make_naive_tensor_view<address_space_enum::global>(
                reinterpret_cast<const ADataType*>(kargs.a_ptr),
                is_a_row_major ? make_tuple(kargs.M, kargs.K) : make_tuple(kargs.K, kargs.M),
                make_tuple(kargs.stride_A, 1),
                number<GemmMxPipeline::GetVectorSizeA()>{},
                number<1>{});
            const auto&& view_m_k = [&]() {
                if constexpr(is_a_row_major)
                    return naive_view;
                else
                    return transform_tensor_view(naive_view,
                                                 make_tuple(make_pass_through_transform(kargs.M),
                                                            make_pass_through_transform(kargs.K)),
                                                 make_tuple(sequence<1>{}, sequence<0>{}),
                                                 make_tuple(sequence<0>{}, sequence<1>{}));
            }();
            const auto pad_view =
                pad_tensor_view(view_m_k,
                                make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                sequence<(is_a_row_major ? false : GemmMxPipeline::kPadM),
                                         (is_a_row_major ? GemmMxPipeline::kPadK : false)>{});
            return make_tile_window(
                pad_view, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), make_tuple(i_m, 0));
        }();
        const auto a_scale_window = [&]() {
            const auto naive_view = make_naive_tensor_view<address_space_enum::global>(
                
            );}()

        const auto b_window = [&]() {
            constexpr bool is_b_col_major =
                std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>;
            const auto naive_view = make_naive_tensor_view<address_space_enum::global>(
                reinterpret_cast<const BDataType*>(kargs.b_ptr),
                is_b_col_major ? make_tuple(kargs.N, kargs.K) : make_tuple(kargs.K, kargs.N),
                make_tuple(kargs.stride_B, 1),
                number<GemmMxPipeline::GetVectorSizeB()>{},
                number<1>{});
            const auto&& view_n_k = [&]() {
                if constexpr(is_b_col_major)
                    return naive_view;
                else
                    return transform_tensor_view(naive_view,
                                                 make_tuple(make_pass_through_transform(kargs.N),
                                                            make_pass_through_transform(kargs.K)),
                                                 make_tuple(sequence<1>{}, sequence<0>{}),
                                                 make_tuple(sequence<0>{}, sequence<1>{}));
            }();
            const auto pad_view =
                pad_tensor_view(view_n_k,
                                make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                sequence<(is_b_col_major ? false : GemmMxPipeline::kPadN),
                                         (is_b_col_major ? GemmMxPipeline::kPadK : false)>{});
            return make_tile_window(
                pad_view, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), make_tuple(i_n, 0));
        }();
        const auto e_window = [&]() {
            constexpr bool is_e_row_major = std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>;
            const auto naive_view         = make_naive_tensor_view<address_space_enum::global>(
                reinterpret_cast<EDataType*>(kargs.c_ptr),
                is_e_row_major ? make_tuple(kargs.M, kargs.N) : make_tuple(kargs.N, kargs.M),
                make_tuple(kargs.stride_C, 1),
                number<EpiloguePipeline::GetVectorSizeC()>{},
                number<1>{});
            const auto&& view_m_n = [&]() {
                if constexpr(is_e_row_major)
                    return naive_view;
                else
                    return transform_tensor_view(naive_view,
                                                 make_tuple(make_pass_through_transform(kargs.M),
                                                            make_pass_through_transform(kargs.N)),
                                                 make_tuple(sequence<1>{}, sequence<0>{}),
                                                 make_tuple(sequence<0>{}, sequence<1>{}));
            }();
            const auto pad_view =
                pad_tensor_view(view_m_n,
                                make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                sequence<(is_e_row_major ? false : GemmMxPipeline::kPadM),
                                         (is_e_row_major ? GemmMxPipeline::kPadN : false)>{});
            return make_tile_window(pad_view,
                                    make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                    make_tuple(i_m, i_n));
        }();

        const index_t num_loop   = TilePartitioner::GetLoopNum(kargs.K);
        const auto& d_window     = make_tuple();
        const auto& c_block_tile = GemmMxPipeline{}(a_window, b_window, num_loop, smem_ptr);
        EpiloguePipeline{}(e_window, c_block_tile, d_window, smem_ptr);
    }

    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {

        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockIdx.x);
        const index_t i_m   = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);
        __shared__ char smem_ptr[GetSmemSize()];
        RunGemmMx(smem_ptr, kargs, i_m, i_n);
    }
};

} // namespace ck_tile
