// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
namespace detail {

template <typename Problem>
struct GemmABQuantPipelineAgBgCrAsyncPolicy
{
    static constexpr auto I0             = number<0>{};
    static constexpr auto I1             = number<1>{};
    static constexpr auto I2             = number<2>{};
    static constexpr auto WGAccessDouble = WGAttrNumAccessEnum::Double;

    using ALayout         = remove_cvref_t<typename Problem::ALayout>;
    using BLayout         = remove_cvref_t<typename Problem::BLayout>;
    using ADataType       = remove_cvref_t<typename Problem::ADataType>;
    using BDataType       = remove_cvref_t<typename Problem::BDataType>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
    static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>, "Wrong!");
    static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>, "Wrong!");
    static_assert(std::is_same_v<ComputeDataType, fp8_t> || std::is_same_v<ComputeDataType, bf8_t>);
    static_assert(std::is_same_v<CDataType, float>);

    using BlockGemmShape = typename Problem::BlockGemmShape;
    using BlockWarps     = typename BlockGemmShape::BlockWarps;
    using WarpTile       = typename BlockGemmShape::WarpTile;

    static constexpr index_t kBlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock  = BlockGemmShape::kM;
    static constexpr index_t NPerBlock  = BlockGemmShape::kN;
    static constexpr index_t KPerBlock  = BlockGemmShape::kK;
    static constexpr index_t WarpTileM  = WarpTile::at(I0);
    static constexpr index_t WarpTileN  = WarpTile::at(I1);
    static constexpr index_t WarpTileK  = WarpTile::at(I2);
    static constexpr index_t MWarpTiles = MPerBlock / WarpTileM;
    static constexpr index_t NWarpTiles = NPerBlock / WarpTileN;
    static constexpr index_t KWarpTiles = KPerBlock / WarpTileK;

    static constexpr index_t NPerBlockBQ = KPerBlock / Problem::BQuantGroupSize::kN;
    static_assert(Problem::AQuantGroupSize::kM == 1 && Problem::AQuantGroupSize::kK == WarpTileK);

    static constexpr index_t warp_size = get_warp_size();
    static constexpr index_t warp_num  = kBlockSize / warp_size;
    static_assert(warp_size == 64, "Wrong!");

    static_assert(sizeof(ADataType) == sizeof(BDataType), "Wrong!");
    static constexpr index_t ElementSize = sizeof(ADataType);
    static constexpr index_t K2          = Problem::VectorLoadSize / ElementSize; // 16
    static constexpr index_t K1          = WarpTile::at(I2) / K2;                 // 8
    static constexpr index_t K0          = KPerBlock / (K1 * K2);
    static_assert(K0 * K1 * K2 == KPerBlock, "Wrong!");

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAQ() { return 1; }
    CK_TILE_HOST_DEVICE static constexpr auto MakeAQDramTileDistribution()
    {
        constexpr index_t M3 = 4;
        constexpr index_t M0 = MWarpTiles;
        constexpr index_t M2 = warp_size / M3 / M0;
        constexpr index_t M1 = MPerBlock / M0 / M3 / M2;

        static_assert(M0 * M1 * M2 * M3 == MPerBlock, "wrong!");

        constexpr index_t R = kBlockSize / warp_size / M1;
        static_assert(kBlockSize == warp_size * M1 * R, "wrong!");

        return make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<R>,
                ck_tile::tuple<ck_tile::sequence<M0, M1, M2, M3>, ck_tile::sequence<KWarpTiles, 1>>,
                ck_tile::tuple<ck_tile::sequence<0, 2, 1>, ck_tile::sequence<1, 1, 1>>,
                ck_tile::tuple<ck_tile::sequence<0, 0, 1>, ck_tile::sequence<2, 0, 3>>,
                ck_tile::sequence<2>,
                ck_tile::sequence<1>>{});
    }
    CK_TILE_HOST_DEVICE static constexpr auto MakeAQLdsBlockDescriptor()
    {
        constexpr index_t M2 = 4;
        constexpr index_t M1 = WarpTileM / M2;
        constexpr index_t M0 = MWarpTiles;
        static_assert(M0 * M1 * M2 == MPerBlock, "wrong!");

        constexpr auto desc_0 =
            make_naive_tensor_descriptor_packed(number_tuple<KWarpTiles, M1, M0, M2>{});

        constexpr auto desc_1 = transform_tensor_descriptor( //
            desc_0,
            make_tuple(make_merge_transform_v3_division_mod(number_tuple<M0, M1, M2>{}),
                       make_merge_transform_v3_division_mod(number_tuple<KWarpTiles>{})),
            make_tuple(sequence<2, 1, 3>{}, sequence<0>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return desc_1;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ() { return 1; }
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        return make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<warp_num, warp_size / NPerBlockBQ>,
                ck_tile::tuple<ck_tile::sequence<NPerBlockBQ>, ck_tile::sequence<KWarpTiles>>,
                ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0>>,
                ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<0, 1>>,
                ck_tile::sequence<2>,
                ck_tile::sequence<0>>{});
    }
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQLdsBlockDescriptor()
    {
        return make_naive_tensor_descriptor_packed(number_tuple<NPerBlockBQ, KWarpTiles>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeAQBlockDistribution()
    {
        return GemmAQuantPipelineAgBgCrDefaultPolicy::MakeAQDramTileDistribution<Problem>();
    }
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQBlockDistribution()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::MakeBQDramTileDistribution<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        static_assert(Problem::BQuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize::kK!");
        static_assert(Problem::TransposeC, "Wrong!");

        using WarpGemm = WarpGemmDispatcher<ComputeDataType,
                                            ComputeDataType,
                                            CDataType,
                                            WarpTileM,
                                            WarpTileN,
                                            WarpTileK,
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            WGAccessDouble>;

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<ADataType,
                                                                      BDataType,
                                                                      CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return ABQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }

    template <index_t MNPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeABDramTileDistribution_()
    {
        constexpr index_t M2 = warp_size / K1;
        constexpr index_t M1 = kBlockSize / warp_size;
        constexpr index_t M0 = MNPerBlock / M1 / M2;
        static_assert(M0 * M1 * M2 == MNPerBlock, "wrong!");

        return make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<M0, M1, M2>, ck_tile::sequence<K0, K1, K2>>,
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>, // M1 M2,K1
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<2, 1>>,
                ck_tile::sequence<1, 2, 2>, // M0,K0,K2
                ck_tile::sequence<0, 0, 2>>{});
    }
    CK_TILE_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        return MakeABDramTileDistribution_<MPerBlock>();
    }
    CK_TILE_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        return MakeABDramTileDistribution_<NPerBlock>();
    }

    template <typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeAsyncLoadDramWindow(const WindowTmp& window_tmp)
    {
        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

        const index_t k_tiles = cols / (K1 * K2);
        const auto col_lens   = make_tuple(k_tiles, number<K1>{}, number<K2>{});

        constexpr index_t M1 = warp_size / static_cast<index_t>(WGAccessDouble) / K1;
        const index_t M0     = integer_divide_ceil(rows, M1);
        const auto row_lens  = make_tuple(M0, number<M1>{});

        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)( // set correct size (without padding)
            d0.get_transforms(),
            tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(number<M1>{}, number<K1>{})),
                       make_pass_through_transform(k_tiles),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc = transform_tensor_descriptor( //
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(row_lens),
                       make_merge_transform_v3_division_mod(col_lens)),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return make_tile_window(make_tensor_view<address_space_enum::global>(
                                    &tensor_view_tmp.get_buffer_view()(0), desc),
                                window_tmp.get_window_lengths(),
                                window_tmp.get_window_origin());
    }

    template <index_t MNPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeABLdsBlockDescriptor_()
    {
        constexpr index_t M3 = warp_size / static_cast<index_t>(WGAccessDouble) / K1;
        constexpr index_t M2 = (warp_size / K1) / M3;
        constexpr index_t M1 = static_cast<index_t>(WGAccessDouble);
        constexpr index_t M0 = MPerBlock / M1 / M2 / M3;

        static_assert(M0 * M1 * M2 * M3 == MPerBlock, "wrong!");

        constexpr index_t PadSize = 16;

        constexpr auto desc_0 = make_naive_tensor_descriptor( //
            number_tuple<M1, K0, M0, M2, M3, K1, K2>{},
            number_tuple<K0 * M0 * M2 * M3 * K1 * K2 + PadSize,
                         M0 * M2 * M3 * K1 * K2,
                         M2 * M3 * K1 * K2,
                         M3 * K1 * K2,
                         K1 * K2,
                         K2,
                         1>{},
            number<K2>{},
            number<1>{});

        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(number<K0>{}),
                       make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<M2>{}),
                       make_xor_transform(make_tuple(number<M3>{}, number<K1>{})),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4, 5>{},
                       sequence<6>{}),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4, 5>{},
                       sequence<6>{}));
        constexpr auto desc_2 = transform_tensor_descriptor( //
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(number_tuple<M0, M1, M2, M3>{}),
                       make_merge_transform_v3_division_mod(number_tuple<K0, K1, K2>{})),
            make_tuple(sequence<2, 0, 3, 4>{}, sequence<1, 5, 6>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return desc_2;
    }
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        return MakeABLdsBlockDescriptor_<MPerBlock>();
    }
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        return MakeABLdsBlockDescriptor_<NPerBlock>();
    }

    CK_TILE_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t desc_size = MakeALdsBlockDescriptor().get_element_space_size();
        return integer_least_multiple(sizeof(typename Problem::ADataType) * desc_size, 16);
    }
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr index_t desc_size = MakeBLdsBlockDescriptor().get_element_space_size();
        return integer_least_multiple(sizeof(typename Problem::BDataType) * desc_size, 16);
    }
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeAQ()
    {
        constexpr index_t desc_size = MakeAQLdsBlockDescriptor().get_element_space_size();
        return sizeof(float) * integer_least_multiple(desc_size, warp_size);
    }
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeBQ()
    {
        constexpr index_t desc_size = MakeBQLdsBlockDescriptor().get_element_space_size();
        return sizeof(float) * integer_least_multiple(desc_size, warp_size);
    }

    CK_TILE_DEVICE static constexpr index_t GetSmemSize()
    {
        return 2 * (GetSmemSizeA() + GetSmemSizeB() + GetSmemSizeAQ() + GetSmemSizeBQ());
    }

    CK_TILE_DEVICE static constexpr auto GetVectorSizeA() { return K2; }
    CK_TILE_DEVICE static constexpr auto GetVectorSizeB() { return K2; }
    CK_TILE_DEVICE static constexpr auto GetSmemPackA() { return K2; }
    CK_TILE_DEVICE static constexpr auto GetSmemPackB() { return K2; }
};
} // namespace detail

struct GemmABQuantPipelineAgBgCrAsyncPolicy
{

#define FORWARD_METHOD_(method)                                               \
    template <typename Problem, typename... Args>                             \
    CK_TILE_HOST_DEVICE static constexpr auto method(Args&&... args)          \
    {                                                                         \
        return detail::GemmABQuantPipelineAgBgCrAsyncPolicy<Problem>::method( \
            std::forward<Args>(args)...);                                     \
    }

    FORWARD_METHOD_(GetVectorSizeAQ);
    FORWARD_METHOD_(MakeAQDramTileDistribution);
    FORWARD_METHOD_(MakeAQLdsBlockDescriptor);
    FORWARD_METHOD_(GetVectorSizeBQ);
    FORWARD_METHOD_(MakeBQDramTileDistribution);
    FORWARD_METHOD_(MakeBQLdsBlockDescriptor);
    FORWARD_METHOD_(MakeAQBlockDistribution);
    FORWARD_METHOD_(MakeBQBlockDistribution);
    FORWARD_METHOD_(GetBlockGemm);
    FORWARD_METHOD_(MakeADramTileDistribution);
    FORWARD_METHOD_(MakeBDramTileDistribution);
    FORWARD_METHOD_(MakeAsyncLoadDramWindow);
    FORWARD_METHOD_(MakeALdsBlockDescriptor);
    FORWARD_METHOD_(MakeBLdsBlockDescriptor);
    FORWARD_METHOD_(GetSmemSizeA);
    FORWARD_METHOD_(GetSmemSizeB);
    FORWARD_METHOD_(GetSmemSizeAQ);
    FORWARD_METHOD_(GetSmemSizeBQ);
    FORWARD_METHOD_(GetSmemSize);
    FORWARD_METHOD_(GetVectorSizeA);
    FORWARD_METHOD_(GetVectorSizeB);
    FORWARD_METHOD_(GetSmemPackA);
    FORWARD_METHOD_(GetSmemPackB);

#undef FORWARD_METHOD_
};

} // namespace ck_tile
