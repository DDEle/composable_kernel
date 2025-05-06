// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_bwd.hpp"
#include "ck_tile/host.hpp"
#include "mask.hpp"
#include "utils.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Convert DQ
using fmha_dtype_0 = FmhaBwdFp16;

// using fmha_bwd_convert_dq_trait_0 = ck_tile::TileFmhaBwdConvertQGradTraits<false, false, 2>;

// using fmha_bwd_convert_dq_pipeline_problem_0 = ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<
//     typename FmhaBwdTypeConfig<fmha_dtype_0>::AccDataType,
//     typename FmhaBwdTypeConfig<fmha_dtype_0>::QGradDataType,
//     /* BlockSize = */ 256,
//     64,
//     128,
//     128,
//     false,
//     false,
//     fmha_bwd_convert_dq_trait_0>;

// using fmha_bwd_convert_dq_0 =
//     typename ck_tile::BlockFmhaBwdConvertQGrad<fmha_bwd_convert_dq_pipeline_problem_0>;

// using fmha_bwd_convert_dq_kernel_0 = ck_tile::FmhaBwdConvertQGradKernel<fmha_bwd_convert_dq_0>;

// using convert_dq_trait_0 = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false,
// false>;

// template <>
// void fmha_bwd_convert_dq_oneshot_<convert_dq_trait_0>(const ck_tile::stream_config& s,
//                                                       fmha_bwd_args a)
// {
//     using k_                               = fmha_bwd_convert_dq_kernel_0;
//     auto [kargs, grids]                    = fmha_bwd_convert_dq_create_kargs_and_grids<k_>(a);
//     constexpr dim3 blocks                  = k_::BlockSize();
//     constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
//     ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{}, grids, blocks, 0, kargs)(
//         ck_tile::stream_config{s.stream_id_});
// }

// template <>
// std::string fmha_bwd_convert_dq_get_name_<convert_dq_trait_0>()
// {
//     using k_ = fmha_bwd_convert_dq_kernel_0;
//     return k_::GetName();
// }

// dq_dk_dv
using fmha_block_tile_0   = ck_tile::sequence<16, 128, 128, 16, 128, 16, 32, 128, 128>;
using fmha_block_warps0_0 = ck_tile::sequence<1, 4, 1>;
using fmha_block_warps1_0 = ck_tile::sequence<4, 1, 1>;
using fmha_block_warps2_0 = ck_tile::sequence<1, 4, 1>;
using fmha_warp_tile0_0   = ck_tile::sequence<16, 16, 32>;
using fmha_warp_tile1_0   = ck_tile::sequence<16, 16, 16>;
using fmha_warp_tile2_0   = ck_tile::sequence<16, 32, 16>;

// TODO: simplify Gemm0~4BlockWarps in TileFmhaBwdShape
//       G0&G2 -> GSdP
//       G1&G3 -> GdKV
//       G4    -> GdQ
using fmha_bwd_shape_0 = ck_tile::TileFmhaBwdShape<fmha_block_tile_0,
                                                   fmha_block_warps0_0,
                                                   fmha_warp_tile0_0,
                                                   fmha_block_warps1_0,
                                                   fmha_warp_tile2_0,
                                                   fmha_block_warps0_0,
                                                   fmha_warp_tile0_0,
                                                   fmha_block_warps1_0,
                                                   fmha_warp_tile2_0,
                                                   fmha_block_warps2_0,
                                                   fmha_warp_tile0_0>;

using fmha_bwd_trait_0 = ck_tile::TileFmhaTraits<true,
                                                 true,
                                                 false,
                                                 false,
                                                 ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                 false,
                                                 false,
                                                 false,
                                                 false,
                                                 1>;
using fmha_mask_0      = ck_tile::SimplifiedGenericAttentionMask<false>;
using fmha_dropout_0   = ck_tile::BlockDropoutBwd<false, true, false>;

using fmha_bwd_pipeline_problem_0 = ck_tile::BlockFmhaBwdPipelineProblem<
    typename FmhaBwdTypeConfig<fmha_dtype_0>::QDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::KDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::VDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::GemmDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::LSEDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::AccDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::DDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::BiasDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::RandValOutputDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::ODataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::OGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::QGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::KGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::VGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype_0>::BiasGradDataType,
    fmha_bwd_shape_0,
    false,
    false,
    fmha_mask_0,
    fmha_dropout_0,
    fmha_bwd_trait_0>;

using fmha_bwd_pipeline_0 =
    ck_tile::BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLP<fmha_bwd_pipeline_problem_0>;

using fmha_bwd_dk_epilogue_0 = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfig<FmhaBwdFp16>::AccDataType,
                                      typename FmhaBwdTypeConfig<FmhaBwdFp16>::KGradDataType,
                                      false,
                                      true>>;

using fmha_bwd_dv_epilogue_0 = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfig<FmhaBwdFp16>::AccDataType,
                                      typename FmhaBwdTypeConfig<FmhaBwdFp16>::VGradDataType,
                                      false,
                                      true>>;

using fmha_bwd_dq_dk_dv_kernel_0 = ck_tile::
    FmhaBwdDQDKDVKernel<fmha_bwd_pipeline_0, fmha_bwd_dk_epilogue_0, fmha_bwd_dv_epilogue_0>;

// using dq_dk_dv_trait_0 = fmha_bwd_dq_dk_dv_traits_<128,
//                                                    FmhaBwdFp16,
//                                                    false,
//                                                    ck_tile::BlockFmhaBwdPipelineEnum::KRKTRVR_IGLP,
//                                                    fmha_mask_0,
//                                                    fmha_dropout_0,
//                                                    ck_tile::BlockAttentionBiasEnum::NO_BIAS,
//                                                    false,
//                                                    true,
//                                                    true,
//                                                    false,
//                                                    false,
//                                                    false>;

// template <>
// void fmha_bwd_dq_dk_dv_oneshot_<dq_dk_dv_trait_0>(const ck_tile::stream_config& s, fmha_bwd_args
// a)
// {
//     using k_                               = fmha_bwd_dq_dk_dv_kernel_0;
//     auto [kargs, grids]                    = fmha_bwd_dq_dk_dv_create_kargs_and_grids<k_>(a);
//     constexpr dim3 blocks                  = k_::BlockSize();
//     constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
//     ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{}, grids, blocks, 0, kargs)(
//         ck_tile::stream_config{s.stream_id_});
// }

// template <>
// std::string fmha_bwd_dq_dk_dv_get_name_<dq_dk_dv_trait_0>()
// {
//     using k_ = fmha_bwd_dq_dk_dv_kernel_0;
//     return k_::GetName();
// }

// // dot_do_o
// using fmha_bwd_dot_do_o_trait_0 = ck_tile::TileFmhaBwdOGradDotOTraits<true, false, 2>;

// using fmha_bwd_dot_do_o_pipeline_problem_0 = ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<
//     typename FmhaBwdTypeConfig<fmha_dtype_0>::ODataType,
//     typename FmhaBwdTypeConfig<fmha_dtype_0>::OGradDataType,
//     typename FmhaBwdTypeConfig<fmha_dtype_0>::DDataType,
//     /* BlockSize = */ 64,
//     128,
//     false,
//     fmha_bwd_dot_do_o_trait_0>;

// using fmha_bwd_dot_do_o_0 =
//     typename ck_tile::BlockFmhaBwdOGradDotO<fmha_bwd_dot_do_o_pipeline_problem_0>;

// using fmha_bwd_dot_do_o_kernel_0 = ck_tile::FmhaBwdOGradDotOKernel<fmha_bwd_dot_do_o_0>;

// using dot_do_o_trait_0 = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;

CK_TILE_DEVICE_EXTERN int obj()
{
    using namespace ck_tile;
    using Policy  = BlockFmhaBwdPipelineDefaultPolicy;
    using Problem = fmha_bwd_pipeline_problem_0;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using KDataType      = remove_cvref_t<typename Problem::KDataType>;
    KDataType* k_lds_ptr = nullptr;
    auto k_lds_read      = make_tensor_view<address_space_enum::lds>(
        k_lds_ptr, Policy::template MakeKLdsReadBlockDescriptor<Problem>());
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    constexpr auto kSeq0                = 64;

    auto k_lds_read_window =
        make_tile_window(k_lds_read,
                         make_tuple(number<kSeq0>{}, number<kQKHeaddim>{}),
                         {0, 0},
                         Policy::template MakeKRegSliceBlockDescriptor<Problem>());

    CK_TILE_PRINT<BlockFmhaShape>();
    // using aaa =
    //     ck_tile::TileFmhaBwdShape<ck_tile::sequence<16, 128, 128, 16, 128, 16, 32, 128, 128>,
    //                               ck_tile::sequence<1, 4, 1>,
    //                               ck_tile::sequence<16, 16, 32>,
    //                               ck_tile::sequence<4, 1, 1>,
    //                               ck_tile::sequence<16, 32, 16>,
    //                               ck_tile::sequence<1, 4, 1>,
    //                               ck_tile::sequence<16, 16, 32>,
    //                               ck_tile::sequence<4, 1, 1>,
    //                               ck_tile::sequence<16, 32, 16>,
    //                               ck_tile::sequence<1, 4, 1>,
    //                               ck_tile::sequence<16, 16, 32>>;
    using k_lds_read_window_t = decltype(k_lds_read_window);

    // array<tuple<WindowAdaptorCoord, BottomTensorCoord>, NumCoord>
    CK_TILE_PRINT<decltype(k_lds_read_window.pre_computed_coords_)>();
    using pre_computed_coords_ = ck_tile::array<
        ck_tile::tuple<ck_tile::tensor_adaptor_coordinate<11,
                                                          ck_tile::sequence<0, 1>,
                                                          ck_tile::sequence<9, 10, 3, 6, 8>>,
                       ck_tile::tensor_coordinate<13, ck_tile::sequence<11, 12>>>,
        1>;

    // /root/ck/include/ck_tile/core/tensor/tile_window.hpp:69
    CK_TILE_PRINT<typename k_lds_read_window_t::WindowAdaptorCoord>();
    using WindowAdaptor = typename k_lds_read_window_t::WindowAdaptor;
    static_assert(std::is_same_v<typename decltype(Policy::template MakeKRegSliceBlockDescriptor<
                                                   Problem>())::PsYs2XsAdaptor,
                                 WindowAdaptor>);
    CK_TILE_PRINT<WindowAdaptor>();
    // using WindowAdaptor_t = ck_tile::tensor_adaptor<
    //     // Transforms
    //     ck_tile::tuple<
    //         ck_tile::replicate<ck_tile::tuple<ck_tile::constant<1>>>,
    //         ck_tile::unmerge<
    //             ck_tile::tuple<ck_tile::constant<1>, ck_tile::constant<4>,
    //             ck_tile::constant<16>>, false>,
    //         ck_tile::unmerge<ck_tile::tuple<ck_tile::constant<4>,
    //                                         ck_tile::constant<2>,
    //                                         ck_tile::constant<4>,
    //                                         ck_tile::constant<4>>,
    //                          false>,
    //         ck_tile::merge_v2_magic_division<
    //             ck_tile::tuple<ck_tile::constant<1>, ck_tile::constant<4>>>,
    //         ck_tile::merge_v2_magic_division<
    //             ck_tile::tuple<ck_tile::constant<4>, ck_tile::constant<16>>>>,
    //     // LowerDimensionHiddenIdss
    //     ck_tile::tuple<ck_tile::sequence<>,
    //                    ck_tile::sequence<0>,
    //                    ck_tile::sequence<1>,
    //                    ck_tile::sequence<2, 4>,
    //                    ck_tile::sequence<8, 5>>,
    //     // UpperDimensionHiddenIdss
    //     ck_tile::tuple<ck_tile::sequence<2>,
    //                    ck_tile::sequence<3, 4, 5>,
    //                    ck_tile::sequence<6, 7, 8, 9>,
    //                    ck_tile::sequence<10>,
    //                    ck_tile::sequence<11>>,
    //     // BottomDimensionHiddenIds
    //     ck_tile::sequence<0, 1>,
    //     // TopDimensionHiddenIds
    //     ck_tile::sequence<10, 11, 3, 6, 7, 9>>;
    CK_TILE_PRINT<typename k_lds_read_window_t::AdaptorTopIndex>();

    using BottomTensorDesc = typename k_lds_read_window_t::BottomTensorDesc;
    static_assert(std::is_same_v<decltype(Policy::template MakeKLdsReadBlockDescriptor<Problem>()),
                                 BottomTensorDesc>);
    CK_TILE_PRINT<BottomTensorDesc>();
    CK_TILE_PRINT<typename BottomTensorDesc::GuaranteedVectorLengths,
                  typename BottomTensorDesc::GuaranteedVectorStrides>();
    CK_TILE_PRINT<decltype(BottomTensorDesc::get_top_dimension_hidden_ids())>();

    // using BottomTensorDesc_t = ck_tile::tensor_descriptor<
    //     // Transforms
    //     ck_tile::tuple<ck_tile::embed<ck_tile::tuple<ck_tile::constant<4>,
    //                                                  ck_tile::constant<2>,
    //                                                  ck_tile::constant<2>,
    //                                                  ck_tile::constant<2>,
    //                                                  ck_tile::constant<2>,
    //                                                  ck_tile::constant<2>,
    //                                                  ck_tile::constant<2>,
    //                                                  ck_tile::constant<2>,
    //                                                  ck_tile::constant<2>,
    //                                                  ck_tile::constant<8>>,
    //                                   ck_tile::tuple<ck_tile::constant<2176>,
    //                                                  ck_tile::constant<1088>,
    //                                                  ck_tile::constant<528>,
    //                                                  ck_tile::constant<256>,
    //                                                  ck_tile::constant<128>,
    //                                                  ck_tile::constant<64>,
    //                                                  ck_tile::constant<32>,
    //                                                  ck_tile::constant<16>,
    //                                                  ck_tile::constant<8>,
    //                                                  ck_tile::constant<1>>>,
    //                    ck_tile::merge_v3_division_mod<ck_tile::tuple<ck_tile::constant<4>,
    //                                                                  ck_tile::constant<2>,
    //                                                                  ck_tile::constant<2>,
    //                                                                  ck_tile::constant<2>,
    //                                                                  ck_tile::constant<2>>>,
    //                    ck_tile::merge_v3_division_mod<ck_tile::tuple<ck_tile::constant<2>,
    //                                                                  ck_tile::constant<2>,
    //                                                                  ck_tile::constant<8>,
    //                                                                  ck_tile::constant<2>,
    //                                                                  ck_tile::constant<2>>>>,
    //     // LowerDimensionHiddenIdss
    //     ck_tile::tuple<ck_tile::sequence<0>,
    //                    ck_tile::sequence<1, 5, 6, 9, 3>,
    //                    ck_tile::sequence<2, 4, 10, 8, 7>>,
    //     // UpperDimensionHiddenIdss
    //     ck_tile::tuple<ck_tile::sequence<1, 2, 3, 4, 5, 6, 7, 8, 9, 10>,
    //                    ck_tile::sequence<11>,
    //                    ck_tile::sequence<12>>,
    //     // TopDimensionHiddenIds
    //     ck_tile::sequence<11, 12>,
    //     // ElementSpaceSize
    //     ck_tile::constant<8656L>,
    //     // GuaranteedVectorLengths_
    //     ck_tile::sequence<-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1>,
    //     // GuaranteedVectorSrides_
    //     ck_tile::sequence<-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1>>;
    CK_TILE_PRINT<typename k_lds_read_window_t::BottomTensorIndex>();

    using load_store_traits = typename k_lds_read_window_t::load_store_traits;
    CK_TILE_PRINT<load_store_traits::NumAccess>();
    CK_TILE_PRINT<decltype(load_store_traits::scalars_per_access_)>();

    constexpr auto tmp0 = BottomTensorDesc::get_top_dimension_safe_vector_length_strides();
    // ck_tile::tuple<ck_tile::array<int, 2>, ck_tile::array<int, 2>>
    CK_TILE_PRINT<tmp0.template get<0>().template get<0>(),
                  tmp0.template get<0>().template get<1>(),
                  tmp0.template get<1>().template get<0>(),
                  tmp0.template get<1>().template get<1>()>();
    CK_TILE_PRINT<ck_tile::constant<BottomTensorDesc::get_num_of_hidden_dimension()>,
                  decltype(WindowAdaptor::get_bottom_dimension_hidden_ids())>();
    CK_TILE_PRINT<k_lds_read_window_t::TileDstr::get_num_of_dimension_p()>();
    CK_TILE_PRINT<k_lds_read_window_t::NDimWindowAdaptorTop>();

    using Traits        = load_store_traits;
    constexpr auto tmp2 = generate_tuple(
        [&](auto i) {
            using SFC_Ys         = typename Traits::SFC_Ys;
            constexpr auto NDimY = k_lds_read_window_t::NDimY;

            constexpr index_t iCoordAccess = i / NDimY;
            constexpr index_t j_           = i % NDimY;
            constexpr index_t j            = j_ * Traits::ScalarPerVector;
            constexpr auto iAccess         = number<iCoordAccess>{};

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

            constexpr auto idx_ys = generate_tuple(
                [&](auto jj) {
                    return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                },
                number<NDimY>{});
            constexpr auto tile_dstr = typename k_lds_read_window_t::TileDstr{};

            constexpr index_t d =
                tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) / Traits::PackedSize;
            // constexpr auto idx_ys = idx_ys_start[number<jj>{}];
            constexpr index_t vec_idx = j / Traits::PackedSize;

            constexpr auto idx_ys_seq       = TO_SEQUENCE(idx_ys, NDimY);
            constexpr auto idx_ys_start_seq = TO_SEQUENCE(idx_ys_start, NDimY);
            return merge_sequences(
                ck_tile::sequence<iCoordAccess, j, d, vec_idx>{}, idx_ys_start_seq, idx_ys_seq);
        },
        number<(Traits::PackedSize / Traits::ScalarPerVector) * Traits::NumAccess>{});
    CK_TILE_PRINT<number<Traits::PackedSize>, number<Traits::ScalarPerVector>, decltype(tmp2)>();
    using tmp3 = ck_tile::tuple<ck_tile::sequence<0, 0, 0   , 0,    0, 0, 0, 0,     0, 0, 0, 0>,
                                ck_tile::sequence<0, 1, 32  , 1,    0, 0, 0, 0,     1, 0, 0, 0>,
                                ck_tile::sequence<0, 2, 64  , 2,    0, 0, 0, 0,     2, 0, 0, 0>,
                                ck_tile::sequence<0, 3, 96  , 3,    0, 0, 0, 0,     3, 0, 0, 0>,
                                ck_tile::sequence<1, 0, 1   , 0,    0, 0, 0, 1,     0, 0, 0, 1>,
                                ck_tile::sequence<1, 1, 33  , 1,    0, 0, 0, 1,     1, 0, 0, 1>,
                                ck_tile::sequence<1, 2, 65  , 2,    0, 0, 0, 1,     2, 0, 0, 1>,
                                ck_tile::sequence<1, 3, 97  , 3,    0, 0, 0, 1,     3, 0, 0, 1>,
                                ck_tile::sequence<2, 0, 2   , 0,    0, 0, 0, 2,     0, 0, 0, 2>,
                                ck_tile::sequence<2, 1, 34  , 1,    0, 0, 0, 2,     1, 0, 0, 2>,
                                ck_tile::sequence<2, 2, 66  , 2,    0, 0, 0, 2,     2, 0, 0, 2>,
                                ck_tile::sequence<2, 3, 98  , 3,    0, 0, 0, 2,     3, 0, 0, 2>,
                                ck_tile::sequence<3, 0, 3   , 0,    0, 0, 0, 3,     0, 0, 0, 3>,
                                ck_tile::sequence<3, 1, 35  , 1,    0, 0, 0, 3,     1, 0, 0, 3>,
                                ck_tile::sequence<3, 2, 67  , 2,    0, 0, 0, 3,     2, 0, 0, 3>,
                                ck_tile::sequence<3, 3, 99  , 3,    0, 0, 0, 3,     3, 0, 0, 3>,
                                ck_tile::sequence<4, 0, 7   , 0,    0, 0, 1, 3,     0, 0, 1, 3>,
                                ck_tile::sequence<4, 1, 39  , 1,    0, 0, 1, 3,     1, 0, 1, 3>,
                                ck_tile::sequence<4, 2, 71  , 2,    0, 0, 1, 3,     2, 0, 1, 3>,
                                ck_tile::sequence<4, 3, 103 , 3,    0, 0, 1, 3,     3, 0, 1, 3>,
                                ck_tile::sequence<5, 0, 6   , 0,    0, 0, 1, 2,     0, 0, 1, 2>,
                                ck_tile::sequence<5, 1, 38  , 1,    0, 0, 1, 2,     1, 0, 1, 2>,
                                ck_tile::sequence<5, 2, 70  , 2,    0, 0, 1, 2,     2, 0, 1, 2>,
                                ck_tile::sequence<5, 3, 102 , 3,    0, 0, 1, 2,     3, 0, 1, 2>,
                                ck_tile::sequence<6, 0, 5   , 0,    0, 0, 1, 1,     0, 0, 1, 1>,
                                ck_tile::sequence<6, 1, 37  , 1,    0, 0, 1, 1,     1, 0, 1, 1>,
                                ck_tile::sequence<6, 2, 69  , 2,    0, 0, 1, 1,     2, 0, 1, 1>,
                                ck_tile::sequence<6, 3, 101 , 3,    0, 0, 1, 1,     3, 0, 1, 1>,
                                ck_tile::sequence<7, 0, 4   , 0,    0, 0, 1, 0,     0, 0, 1, 0>,
                                ck_tile::sequence<7, 1, 36  , 1,    0, 0, 1, 0,     1, 0, 1, 0>,
                                ck_tile::sequence<7, 2, 68  , 2,    0, 0, 1, 0,     2, 0, 1, 0>,
                                ck_tile::sequence<7, 3, 100 , 3,    0, 0, 1, 0,     3, 0, 1, 0>>;

    return 0;
}

int main(int argc, char* argv[]) { return 0; }
