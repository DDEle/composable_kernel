// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "mask.hpp"
#include "bias.hpp"

#include <type_traits>
#include <utility>
#include <variant>

struct FmhaBwdFp16
{
};

struct FmhaBwdBf16
{
};

template <typename DataType>
struct FmhaBwdTypeConfig;

template <>
struct FmhaBwdTypeConfig<FmhaBwdFp16>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using GemmDataType          = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::half_t;
    using OGradDataType         = ck_tile::half_t;
    using QGradDataType         = ck_tile::half_t;
    using KGradDataType         = ck_tile::half_t;
    using VGradDataType         = ck_tile::half_t;
    using BiasGradDataType      = ck_tile::half_t;
};

template <>
struct FmhaBwdTypeConfig<FmhaBwdBf16>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using GemmDataType          = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using LSEDataType           = float;
    using AccDataType           = float; // data type for gemm accumulation
    using DDataType             = float;
    using RandValOutputDataType = uint8_t;
    using ODataType             = ck_tile::bf16_t;
    using OGradDataType         = ck_tile::bf16_t;
    using QGradDataType         = ck_tile::bf16_t;
    using KGradDataType         = ck_tile::bf16_t;
    using VGradDataType         = ck_tile::bf16_t;
    using BiasGradDataType      = ck_tile::bf16_t;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

// runtime args, some will passed to karg, some will used to compute grids/blocks
struct fmha_bwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* o_ptr;
    const void* lse_ptr;
    const void* do_ptr;
    void* d_ptr;
    void* rand_val_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* dbias_ptr;
    void* dq_acc_ptr;
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t max_seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    float scale;
    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o;
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_do;
    ck_tile::index_t stride_dq_acc;
    ck_tile::index_t stride_dq;
    ck_tile::index_t stride_dk;
    ck_tile::index_t stride_dv;
    ck_tile::index_t stride_dbias;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_lsed;
    ck_tile::index_t nhead_stride_dq_acc;
    ck_tile::index_t nhead_stride_dq;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t nhead_stride_dbias;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_do;
    ck_tile::index_t batch_stride_lsed;
    ck_tile::index_t batch_stride_dq_acc;
    ck_tile::index_t batch_stride_dq;
    ck_tile::index_t batch_stride_dk;
    ck_tile::index_t batch_stride_dv;
    ck_tile::index_t batch_stride_dbias;
    ck_tile::index_t split_stride_dq_acc;
    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    float p_drop;
    float p_undrop;
    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

template <typename FmhaBwdDQDKDVKernel>
auto fmha_bwd_dq_dk_dv_create_kargs_and_grids(fmha_bwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdDQDKDVKernel::kIsGroupMode)
        {
            return FmhaBwdDQDKDVKernel::MakeKargsImpl(args.q_ptr,
                                                      args.k_ptr,
                                                      args.v_ptr,
                                                      args.bias_ptr,
                                                      args.lse_ptr,
                                                      args.do_ptr,
                                                      args.d_ptr,
                                                      args.rand_val_ptr,
                                                      args.dk_ptr,
                                                      args.dv_ptr,
                                                      args.dbias_ptr,
                                                      args.dq_acc_ptr,
                                                      args.seqstart_q_ptr,
                                                      args.seqstart_k_ptr,
                                                      args.seqlen_k_ptr,
                                                      args.hdim_q,
                                                      args.hdim_v,
                                                      args.nhead_q,
                                                      args.nhead_q / args.nhead_k,
                                                      args.scale,
                                                      args.stride_q,
                                                      args.stride_k,
                                                      args.stride_v,
                                                      args.stride_bias,
                                                      args.stride_randval,
                                                      args.stride_do,
                                                      args.stride_dq_acc,
                                                      args.stride_dk,
                                                      args.stride_dv,
                                                      args.stride_dbias,
                                                      args.nhead_stride_q,
                                                      args.nhead_stride_k,
                                                      args.nhead_stride_v,
                                                      args.nhead_stride_bias,
                                                      args.nhead_stride_randval,
                                                      args.nhead_stride_do,
                                                      args.nhead_stride_lsed,
                                                      args.nhead_stride_dq_acc,
                                                      args.nhead_stride_dk,
                                                      args.nhead_stride_dv,
                                                      args.nhead_stride_dbias,
                                                      args.split_stride_dq_acc,
                                                      args.window_size_left,
                                                      args.window_size_right,
                                                      args.mask_type,
                                                      args.p_drop,
                                                      args.drop_seed_offset);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdDQDKDVKernel::MakeKargsImpl(args.q_ptr,
                                                      args.k_ptr,
                                                      args.v_ptr,
                                                      args.bias_ptr,
                                                      args.lse_ptr,
                                                      args.do_ptr,
                                                      args.d_ptr,
                                                      args.rand_val_ptr,
                                                      args.dk_ptr,
                                                      args.dv_ptr,
                                                      args.dbias_ptr,
                                                      args.dq_acc_ptr,
                                                      args.seqlen_q,
                                                      args.seqlen_k,
                                                      args.hdim_q,
                                                      args.hdim_v,
                                                      args.nhead_q,
                                                      args.nhead_q / args.nhead_k,
                                                      args.scale,
                                                      args.stride_q,
                                                      args.stride_k,
                                                      args.stride_v,
                                                      args.stride_bias,
                                                      args.stride_randval,
                                                      args.stride_do,
                                                      args.stride_dq_acc,
                                                      args.stride_dk,
                                                      args.stride_dv,
                                                      args.stride_dbias,
                                                      args.nhead_stride_q,
                                                      args.nhead_stride_k,
                                                      args.nhead_stride_v,
                                                      args.nhead_stride_bias,
                                                      args.nhead_stride_randval,
                                                      args.nhead_stride_do,
                                                      args.nhead_stride_lsed,
                                                      args.nhead_stride_dq_acc,
                                                      args.nhead_stride_dk,
                                                      args.nhead_stride_dv,
                                                      args.nhead_stride_dbias,
                                                      args.batch_stride_q,
                                                      args.batch_stride_k,
                                                      args.batch_stride_v,
                                                      args.batch_stride_bias,
                                                      args.batch_stride_randval,
                                                      args.batch_stride_do,
                                                      args.batch_stride_lsed,
                                                      args.batch_stride_dq_acc,
                                                      args.batch_stride_dk,
                                                      args.batch_stride_dv,
                                                      args.batch_stride_dbias,
                                                      args.split_stride_dq_acc,
                                                      args.window_size_left,
                                                      args.window_size_right,
                                                      args.mask_type,
                                                      args.p_drop,
                                                      args.drop_seed_offset);
        }
    }();

    dim3 grids = FmhaBwdDQDKDVKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_k);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaBwdOGradDotOKernel>
auto fmha_bwd_dot_do_o_create_kargs_and_grids(fmha_bwd_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdOGradDotOKernel::kIsGroupMode)
        {
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.p_undrop,
                                                     args.seqstart_q_ptr,
                                                     args.hdim_v,
                                                     args.stride_do,
                                                     args.stride_o,
                                                     args.nhead_stride_do,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_lsed);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdOGradDotOKernel::MakeKargs(args.o_ptr,
                                                     args.do_ptr,
                                                     args.d_ptr,
                                                     args.p_undrop,
                                                     args.seqlen_q,
                                                     args.hdim_v,
                                                     args.stride_do,
                                                     args.stride_o,
                                                     args.nhead_stride_do,
                                                     args.nhead_stride_o,
                                                     args.nhead_stride_lsed,
                                                     args.batch_stride_do,
                                                     args.batch_stride_o,
                                                     args.batch_stride_lsed);
        }
    }();

    dim3 grids = FmhaBwdOGradDotOKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

template <typename FmhaBwdConvertQGradKernel>
auto fmha_bwd_convert_dq_create_kargs_and_grids(fmha_bwd_args args)
{
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaBwdConvertQGradKernel::kIsGroupMode)
        {
            return FmhaBwdConvertQGradKernel::MakeKargs(args.dq_acc_ptr,
                                                        args.dq_ptr,
                                                        args.seqstart_q_ptr,
                                                        args.seqstart_k_ptr,
                                                        args.hdim_q,
                                                        args.stride_dq,
                                                        args.stride_dq_acc,
                                                        args.nhead_stride_dq,
                                                        args.nhead_stride_dq_acc,
                                                        args.split_stride_dq_acc);
        }
        else
        { // create batch mode kernel arguments
            return FmhaBwdConvertQGradKernel::MakeKargs(args.dq_acc_ptr,
                                                        args.dq_ptr,
                                                        args.seqlen_q,
                                                        args.seqlen_k,
                                                        args.hdim_q,
                                                        args.stride_dq,
                                                        args.stride_dq_acc,
                                                        args.nhead_stride_dq,
                                                        args.nhead_stride_dq_acc,
                                                        args.batch_stride_dq,
                                                        args.batch_stride_dq_acc,
                                                        args.split_stride_dq_acc);
        }
    }();

    dim3 grids = FmhaBwdConvertQGradKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q);
    return ck_tile::make_tuple(kargs, grids);
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename FmhaDropout_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kPadD_,
          bool kPadDv_,
          bool kIsDeterministic_,
          bool kUseTrLoad_>
struct fmha_bwd_dq_dk_dv_traits_
{
    static constexpr bool is_fmha_bwd_dq_dk_dv_traits_ = true;

    static constexpr ck_tile::index_t HDim = HDim_;
    using DataType                         = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    using FmhaMask                         = ck_tile::remove_cvref_t<FmhaMask_>;
    using FmhaDropout                      = ck_tile::remove_cvref_t<FmhaDropout_>;
    static constexpr auto BiasEnum         = BiasEnum_;
    static constexpr bool kHasBiasGrad     = kHasBiasGrad_;
    static constexpr bool kPadD            = kPadD_;
    static constexpr bool kPadDv           = kPadDv_;
    static constexpr bool kIsDeterministic = kIsDeterministic_;
    static constexpr bool kUseTrLoad       = kUseTrLoad_;
};
template <class T>
concept fmha_bwd_dq_dk_dv_traits_c = requires { static_assert(T::is_fmha_bwd_dq_dk_dv_traits_); };

template <typename Traits_>
float fmha_bwd_dq_dk_dv_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_>
void fmha_bwd_dq_dk_dv_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_>
std::string fmha_bwd_dq_dk_dv_get_name_();

template <ck_tile::index_t HDim_, typename DataType_, bool kIsGroupMode_, bool kPadS_, bool kPadDv_>
struct fmha_bwd_dot_do_o_traits_
{
    static constexpr bool is_fmha_bwd_dot_do_o_traits_ = true;

    static constexpr ck_tile::index_t HDim = HDim_;
    using DataType                         = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    static constexpr bool kPadS            = kPadS_;
    static constexpr bool kPadDv           = kPadDv_;
};
template <class T>
concept fmha_bwd_dot_do_o_traits_c = requires { static_assert(T::is_fmha_bwd_dot_do_o_traits_); };

template <typename Traits_>
float fmha_bwd_dot_do_o_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_>
void fmha_bwd_dot_do_o_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_>
std::string fmha_bwd_dot_do_o_get_name_();

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          bool kPadS_,
          bool kPadD_,
          bool kIsDeterministic_>
struct fmha_bwd_convert_dq_traits_
{
    static constexpr bool is_fmha_bwd_convert_dq_traits_ = true;

    static constexpr ck_tile::index_t HDim = HDim_;
    using DataType                         = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    static constexpr bool kPadS            = kPadS_;
    static constexpr bool kPadD            = kPadD_;
    static constexpr bool kIsDeterministic = kIsDeterministic_;
};
template <class T>
concept fmha_bwd_convert_dq_traits_c =
    requires { static_assert(T::is_fmha_bwd_convert_dq_traits_); };

template <typename Traits_>
float fmha_bwd_convert_dq_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_>
void fmha_bwd_convert_dq_oneshot_(const ck_tile::stream_config&, fmha_bwd_args);

template <typename Traits_>
std::string fmha_bwd_convert_dq_get_name_();

// This is the public API, will be generated by script
struct fmha_bwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_dbias;
    bool has_dropout;
    bool is_store_randval;
    bool is_deterministic;
    // TODO: padding check is inside this api
};
template <int Version = 2>
float fmha_bwd(fmha_bwd_traits, fmha_bwd_args, const ck_tile::stream_config&);

struct FmhaBwdTileSize
{
    index_t kBM0;      // tile size along q seqlen (block size)
    index_t kBN0;      // tile size along k seqlen
    index_t kBK0;      // tile size along gemm0 unroll(F_bhdq)
    index_t kBK1;      // tile size along gemm1 unroll(F_bm0)
    index_t kBK2;      // tile size along gemm2 unroll(F_bhdv)
    index_t kBK3;      // tile size along gemm3 unroll(F_bm0)
    index_t kBK4;      // tile size along gemm4 unroll(F_bn0)
    index_t kBHDQ;     // q head_dim
    index_t kBHDV;     // v head_dim
    index_t kRM0;      // number of warps along q seqlen (block warps) in gemm0/gemm2
    index_t kRN0;      // number of warps along k seqlen (block warps) in gemm0/gemm2
    index_t kRK0;      // number of warps along headdim_qk/v (not used) in gemm0/gemm2
    index_t kRM1;      // number of warps along k seqlen (block warps) in gemm1/gemm3
    index_t kRN1;      // number of warps along headdim_qk/v (block warps) in gemm1/gemm3
    index_t kRK1;      // number of warps along q seqlen (not used) in gemm1/gemm3
    index_t kRM2;      // number of warps along q seqlen (block warps) in gemm4
    index_t kRN2;      // number of warps along headdim_qk (block warps) in gemm4
    index_t kRK2;      // number of warps along k seqlen (not used) in gemm4
    index_t kWM0;      // warp size along m in gemm0/gemm2/gemm4
    index_t kWN0;      // warp size along n in gemm0/gemm2/gemm4
    index_t kWK0;      // warp size along k in gemm0/gemm2/gemm4
    index_t kWM1;      // warp size along m in gemm1/gemm3
    index_t kWN1;      // warp size along n in gemm1/gemm3
    index_t kWK1;      // warp size along k in gemm1/gemm3
    index_t occupancy; // occupancy
};
consteval FmhaBwdTileSize get_fmha_bwd_tile_size(ck_tile::index_t hdim, bool tr_load)
{
    if(tr_load)
    {
        switch(hdim)
        {
        case 128:
            return FmhaBwdTileSize{32, 128, 128, 32, 128, 32, 32, 128, 128, 1,  4,  1, 4,
                                   1,  1,   1,   4,  1,   16, 16, 32,  16,  16, 32, 1};
        default: throw std::runtime_error("Unsupported hdim for fmha_bwd tile size with tr_load");
        }
    }
    else
    {
        switch(hdim)
        {
        case 32:
            return FmhaBwdTileSize{32, 128, 32, 32, 32, 32, 64, 32, 32, 1,  4,  1, 4,
                                   1,  1,   2,  2,  1,  16, 16, 32, 16, 16, 16, 1};
        case 64:
            return FmhaBwdTileSize{32, 128, 64, 32, 64, 32, 32, 64, 64, 1,  4,  1, 4,
                                   1,  1,   1,  4,  1,  16, 16, 32, 16, 16, 16, 1};
        case 128:
            return FmhaBwdTileSize{16, 128, 128, 16, 128, 16, 32, 128, 128, 1,  4,  1, 4,
                                   1,  1,   1,   4,  1,   16, 16, 32,  16,  16, 16, 1};
        case 160:
            return FmhaBwdTileSize{32, 64, 160, 32, 160, 32, 32, 160, 160, 1,  4,  1, 4,
                                   1,  1,  2,   2,  1,   16, 16, 32,  16,  16, 16, 1};
        case 256:
            return FmhaBwdTileSize{16, 64, 256, 16, 256, 16, 32, 256, 256, 1,  4,  1, 4,
                                   1,  1,  1,   4,  1,   16, 16, 32,  16,  16, 16, 1};
        default:
            throw std::runtime_error("Unsupported hdim for fmha_bwd tile size without tr_load");
        }
    }
}

template <typename Traits>
struct FmhaBwdKernelImpl
{
    static constexpr std::string GetName();
    static void Run(const ck_tile::stream_config& s, fmha_bwd_args a);
};

enum class fmha_bwd_kernel_kind
{
    dot_do_o,
    dq_dk_dv,
    convert_dq,
};

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          typename FmhaMask_,
          typename FmhaDropout_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kHasBiasGrad_,
          bool kPadS1D_,
          bool kPadD_,
          bool kPadDv_,
          bool kIsDeterministic_,
          bool kUseTrLoad_>
class fmha_bwd_kernal_group_traits
{
    template <fmha_bwd_kernel_kind kid>
    struct kernel_traits
    {
        using type = void;
    } template <>
    struct kernel_traits<fmha_bwd_kernel_kind::dot_do_o>
    {
        using type = fmha_bwd_dot_do_o_traits_< //
            HDim_,
            DataType_,
            kIsGroupMode_,
            kPadS1D_,
            kPadDv_>;
    };
    template <>
    struct kernel_traits<fmha_bwd_kernel_kind::dq_dk_dv>
    {
        using type = fmha_bwd_dq_dk_dv_traits_< //
            HDim_,
            DataType_,
            kIsGroupMode_,
            FmhaMask_,
            FmhaDropout_,
            BiasEnum_,
            kHasBiasGrad_,
            kPadS1D_,
            kPadD_,
            kPadDv_,
            kIsDeterministic_,
            kUseTrLoad_>;
    };
    template <>
    struct kernel_traits<fmha_bwd_kernel_kind::convert_dq>
    {
        using type = fmha_bwd_convert_dq_traits_< //
            HDim_,
            DataType_,
            kIsGroupMode_,
            kPadS1D_,
            kPadD_,
            kIsDeterministic_>;
    };

    public:
    static constexpr bool is_fmha_bwd_kernal_group_traits_ = true;
    template <fmha_bwd_kernel_kind kid>
    using kernel_traits_t = typename kernel_traits<kid>::type;
};
template <class T>
concept fmha_bwd_kernal_group_traits_c =
    requires { static_assert(T::is_fmha_bwd_kernal_group_traits_); };

template <fmha_bwd_kernal_group_traits_c T, fmha_bwd_kernel_kind... kid>
class FmhaBwdKernelGroup
{
    static constexpr size_t n_kernels = sizeof...(kid);
    template <size_t I>
    using kid_i = std::tuple_element_t<I, std::tuple<kid...>>;
    public

    static consteval string get_kernel_names()
    {
        std::string kernel_names =
            FmhaBwdKernelImpl<typename T::kernel_traits_t<kid_i<0>>>::GetName();
        for(size_t i = 1; i < T::n_kernels; ++i)
        {
            kernel_names +=
                "@" + FmhaBwdKernelImpl<typename T::kernel_traits_t<kid_i<i>>>::GetName();
        }
        return kernel_names;
    }

    static float run(const ck_tile::stream_config& s, fmha_bwd_args a)
    {

        if(s.log_level_ > 0)
            std::cout << ", " << get_kernel_names() << std::flush;
        return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) {
            (FmhaBwdKernelImpl<typename T::kernel_traits_t<kid>>::Run(s_, a));
        }...);
    }
}
