// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

struct GemmABQuantPipelineAgBgCrDefaultPolicy : public UniversalGemmPipelineAgBgCrPolicy
{
    using Base = UniversalGemmPipelineAgBgCrPolicy;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAQ()
    {
        return GemmAQuantPipelineAgBgCrDefaultPolicy::GetVectorSizeAQ<Problem>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAQDramTileDistribution()
    {
        return GemmAQuantPipelineAgBgCrDefaultPolicy::MakeAQDramTileDistribution<Problem>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::GetVectorSizeBQ<Problem>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::MakeBQDramTileDistribution<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        static_assert(Problem::BQuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize::kK!");
        static_assert(Problem::TransposeC, "Wrong!");

        using WarpGemm = WarpGemmDispatcher<typename Problem::ComputeDataType,
                                            typename Problem::ComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            WGAttrNumAccessEnum::Double>;
        static_assert(std::is_same_v<typename Problem::ComputeDataType, fp8_t> ||
                      std::is_same_v<typename Problem::ComputeDataType, bf8_t>);
        static_assert(std::is_same_v<typename Problem::CDataType, float>);

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return ABQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t MPerBlock  = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock  = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = Problem::GetAlignmentA();
        constexpr index_t K0 = KPerBlock / K1;

        constexpr index_t M2 = get_warp_size() / K0;
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = MPerBlock / M1 / M2;

        static_assert(K0 * K1 == KPerBlock, "wrong!");
        static_assert(M0 * M1 * M2 == MPerBlock, "wrong!");

        return make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<M0, M1, M2>, ck_tile::sequence<K0, K1>>,
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>, // M1 M2,K0
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<2, 0>>,
                ck_tile::sequence<1, 2>, // M0,K1
                ck_tile::sequence<0, 1>>{});
    }
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t NPerBlock  = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock  = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = Problem::GetAlignmentB();
        constexpr index_t K0 = KPerBlock / K1;

        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = NPerBlock / N1 / N2;

        static_assert(K0 * K1 == KPerBlock, "wrong!");
        static_assert(N0 * N1 * N2 == NPerBlock, "wrong!");

        return make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<N0, N1, N2>, ck_tile::sequence<K0, K1>>,
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>, // N1 N2,K0
                ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<2, 0>>,
                ck_tile::sequence<1, 2>, // N0,K1
                ck_tile::sequence<0, 1>>{});
    }

    template <index_t MPerBlock, index_t KPerBlock, index_t KPack>
    CK_TILE_DEVICE static constexpr auto MakeABLdsBlockDescriptor_()
    {
        constexpr index_t K1 = KPack;
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M3 =
            get_warp_size() / static_cast<index_t>(WGAttrNumAccessEnum::Double) / K0;
        constexpr index_t M2 = (get_warp_size() / K0) / M3;
        constexpr index_t M1 = static_cast<index_t>(WGAttrNumAccessEnum::Double);
        constexpr index_t M0 = MPerBlock / M1 / M2 / M3;

        static_assert(K0 * K1 == KPerBlock, "wrong!");
        static_assert(M0 * M1 * M2 * M3 == MPerBlock, "wrong!");

        constexpr index_t PadSize = 16;

        constexpr auto desc_0 = make_naive_tensor_descriptor(
            make_tuple(
                number<M1>{}, number<M0>{}, number<M2>{}, number<M3>{}, number<K0>{}, number<K1>{}),
            make_tuple(number<M0 * M2 * M3 * K0 * K1 + PadSize>{},
                       number<M2 * M3 * K0 * K1>{},
                       number<M3 * K0 * K1>{},
                       number<K0 * K1>{},
                       number<K1>{},
                       number<1>{}),
            number<K1>{},
            number<1>{});

        constexpr auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(number<M1>{}),
                       make_pass_through_transform(number<M0>{}),
                       make_pass_through_transform(number<M2>{}),
                       make_xor_transform(make_tuple(number<M3>{}, number<K0>{})),
                       make_pass_through_transform(number<K1>{})),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3, 4>{}, sequence<5>{}),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3, 4>{}, sequence<5>{}));
        constexpr auto desc_2 = transform_tensor_descriptor( //
            desc_1,
            make_tuple(
                make_merge_transform_v3_division_mod(
                    make_tuple(number<M0>{}, number<M1>{}, number<M2>{}, number<M3>{})),
                make_merge_transform_v3_division_mod(make_tuple(number<K0>{}, number<K1>{}))),
            make_tuple(sequence<1, 0, 2, 3>{}, sequence<4, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return desc_2;
    }

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using ALayout = remove_cvref_t<typename Problem::ALayout>;
        if constexpr(is_a_load_tr<Problem> || get_warp_size() != 64 ||
                     std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
            return Base::MakeALdsBlockDescriptor<Problem, OverrideADataType>();
        return MakeABLdsBlockDescriptor_<Problem::BlockGemmShape::kM,
                                         Problem::BlockGemmShape::kK,
                                         GetSmemPackA<Problem>()>();
    }
    template <typename Problem,
              typename OverrideBDataType = remove_cvref_t<typename Problem::BDataType>>
    CK_TILE_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        using BLayout = remove_cvref_t<typename Problem::BLayout>;
        if constexpr(is_b_load_tr<Problem> || get_warp_size() != 64 ||
                     std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Base::MakeBLdsBlockDescriptor<Problem, OverrideBDataType>();
        return MakeABLdsBlockDescriptor_<Problem::BlockGemmShape::kN,
                                         Problem::BlockGemmShape::kK,
                                         GetSmemPackB<Problem>()>();
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeA()
    {
        return integer_least_multiple(
            sizeof(typename Problem::ADataType) *
                MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
            16);
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeB()
    {
        return integer_least_multiple(
            sizeof(typename Problem::BDataType) *
                MakeBLdsBlockDescriptor<Problem>().get_element_space_size(),
            16);
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetSmemSize()
    {
        return GetSmemSizeA<Problem>() + GetSmemSizeB<Problem>();
    }
};

} // namespace ck_tile
