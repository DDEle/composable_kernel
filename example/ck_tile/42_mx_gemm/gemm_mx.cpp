// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>

#include "ck_tile/host.hpp"
#include "mx_gemm.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename GemmConfig,
          typename TypeConfig,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout>
float invoke_gemm_mx(ck_tile::DeviceMem& a_dev_buf,
                     ck_tile::DeviceMem& b_shuffle_dev_buf,
                     ck_tile::DeviceMem& c_dev_buf,
                     ck_tile::index_t M,
                     ck_tile::index_t N,
                     ck_tile::index_t K,
                     ck_tile::index_t stride_A,
                     ck_tile::index_t stride_B,
                     ck_tile::index_t stride_C,
                     ck_tile::index_t kbatch,
                     ScaleA scale_a,
                     ScaleB scale_b,
                     int n_warmup,
                     int n_repeat)
{
    ck_tile::MxGemmHostArgs args = {a_dev_buf.GetDeviceBuffer(),
                                    b_shuffle_dev_buf.GetDeviceBuffer(),
                                    {},
                                    c_dev_buf.GetDeviceBuffer(),
                                    kbatch,
                                    M,
                                    N,
                                    K,
                                    stride_A,
                                    stride_B,
                                    {},
                                    stride_C,
                                    scale_a,
                                    scale_b};

    using FlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                           GemmConfig::kPadN,
                                           GemmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           CLayout,
                                           GemmConfig::NumWaveGroups>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, FlatmmShape, Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t k_split     = (K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(k_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time = BaseGemmPipeline::template TailHandler<true>(
        [&](auto has_hot_loop_, auto tail_num_) {
            constexpr auto has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_num_v     = tail_num_.value;
            auto invoke_splitk_path       = [&](auto split_k_) {
                return mx_gemm_calc<GemmConfig,
                                          ADataType,
                                          BDataType,
                                          DsDatatype,
                                          AccDataType,
                                          CDataType,
                                          ALayout,
                                          BLayout,
                                          DsLayout,
                                          CLayout,
                                          ScaleA,
                                          ScaleB,
                                          UsePersistentKernel,
                                          CDEElementWise,
                                          split_k_.value,
                                          has_hot_loop_v,
                                          tail_num_v>(
                    args,
                    ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});
            };
            return (args.k_batch == 1) ? invoke_splitk_path(std::false_type{})
                                       : invoke_splitk_path(std::true_type{});
        },
        has_hot_loop,
        tail_num);

    constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

    std::size_t flop     = std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / 32;
    std::size_t num_byte = sizeof(ADataType) * M * K / APackedSize +
                           sizeof(BDataType) * N * K / BPackedSize + sizeof(CDataType) * M * N +
                           sizeof(ck_tile::e8m0_t) * M * K / 32 +
                           sizeof(ck_tile::e8m0_t) * N * K / 32;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run MXFP4_Flatmm kernel " //
              << " M = " << M << " N = " << N << " K = " << K << " StrideA = " << stride_A
              << " StrideB = " << stride_B << " StrideC = " << stride_C << " : " << ave_time
              << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "32", "m dimension")
        .insert("n", "128", "n dimension")
        .insert("k", "256", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert(
            "mx_prec", "fp4xfp4", "data type for activation and weight, support: fp6xfp6, fp8xfp8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:constant(1)")
        .insert("persistent", "0", "0: no persistent, 1: persistent kernel")
        .insert("warp_tile",
                "0",
                "0: 16x16, 1: 32x32, 2: 16x16x128 (950 only), 3: 32x32x64 (950 only)");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <class GemmConfig, class IterSrc, class IterDst>
void preShuffleWeight(const IterSrc src, IterDst dst, int N, int K)
{
    int KPack = 16;
    int NLane = GemmConfig::N_Warp_Tile;
    int KLane = 64 / NLane;
    int K_pk  = K / 2;
    int K0    = K_pk / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K_pk; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K_pk + k];
        }
    }
}

template <class GemmConfig, bool KLast, typename Src>
auto preShuffleScale(Src& src)
{
    using dtype      = typename Src::Data::value_type;
    auto src_lengths = src.get_lengths();
    const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
    const auto K     = KLast ? src_lengths[1] : src_lengths[0];

    size_t MNXdlPack   = 2;
    size_t KXdlPack    = 2;
    size_t XdlMNThread = GemmConfig::N_Warp_Tile; // 16
    size_t XdlKThread  = 64 / XdlMNThread;

    const auto MN_Paded = ck_tile::integer_least_multiple(MN, XdlMNThread * MNXdlPack);

    ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({MN_Paded * K}, {1}));

    size_t K0 = K / KXdlPack / XdlKThread; // KRepeat

    // The 4 16x128 building blocks will be packed into 1 32x256 for F4
    // The 8 16x16x128 mfma will be packed into 1 32x32x256 for F4

    // unfold the MN32xK(256/32) scale buffer
    //    4            16             2           2
    // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
    // Then, MNRepeat->KRepeat

    for(size_t n = 0; n < MN_Paded; ++n)
    {
        for(size_t k = 0; k < K; ++k)
        {
            auto n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
            auto tempn = n % (XdlMNThread * MNXdlPack);
            auto n1    = tempn % XdlMNThread; // i XdlMNThread
            auto n2    = tempn / XdlMNThread; // i MNXdlPack

            auto k0    = k / (XdlKThread * KXdlPack); // i KRepeat
            auto tempk = k % (XdlKThread * KXdlPack);
            auto k1    = tempk % XdlKThread; // i XdlKThread
            auto k2    = tempk / XdlKThread; // i KXdlPack

            auto outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                               k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                               k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                               k2 * MNXdlPack + n2;

            if constexpr(KLast)
                shuffled(outputIndex) = n < MN ? src(n, k) : dtype{};
            else
                shuffled(outputIndex) = n < MN ? src(k, n) : dtype{};
        }
    }
    return shuffled;
}

template <typename PrecActType,
          typename PrecWeightType,
          typename CDataType,
          typename GemmConfig,
          bool UsePersistentKernel = false,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_gemm_mx_with_layouts(int argc,
                             char* argv[],
                             const ALayout a_layout = ALayout{},
                             const BLayout b_layout = BLayout{},
                             const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using ADataType   = PrecActType;
    using BDataType   = PrecWeightType;
    using AccDataType = float;

    using ScaleType = ck_tile::e8m0_t;

    constexpr int ScaleGranularityM = 1;
    constexpr int ScaleGranularityN = 1;
    constexpr int ScaleGranularityK = 32;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    ck_tile::index_t n_warmup    = arg_parser.get_int("warmup");
    ck_tile::index_t n_repeat    = arg_parser.get_int("repeat");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(c_layout));

    auto scale_stride_A = ck_tile::get_default_stride(
        M / ScaleGranularityM, K / ScaleGranularityK, 0, is_row_major(a_layout));
    auto scale_stride_B = ck_tile::get_default_stride(
        K / ScaleGranularityK, N / ScaleGranularityN, 0, is_row_major(b_layout));

    if(K % ScaleGranularityK != 0)
        throw std::runtime_error("wrong! K must be multiple of ScaleGranularityK.");
    if(K % ck_tile::numeric_traits<ADataType>::PackedSize != 0 ||
       K % ck_tile::numeric_traits<BDataType>::PackedSize != 0)
        throw std::runtime_error("wrong! K must be multiple of packed size.");

    ck_tile ::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_origin_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_rslt_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    ck_tile::HostTensor<ScaleType> scale_a(ck_tile::host_tensor_descriptor(
        M / ScaleGranularityM, K / ScaleGranularityK, scale_stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<ScaleType> scale_b(ck_tile::host_tensor_descriptor(
        K / ScaleGranularityK, N / ScaleGranularityN, scale_stride_B, is_row_major(b_layout)));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_origin_host);
        ck_tile::FillUniformDistribution<ScaleType>{-2.f, 2.f}(scale_a);
        ck_tile::FillUniformDistribution<ScaleType>{-2.f, 2.f}(scale_b);
    }
    else if(init_method == 1)
    {
#if 0
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_origin_host);
        ck_tile::FillUniformDistribution<ScaleType>{1.f, 1.f}(scale_a);
        ck_tile::FillUniformDistribution<ScaleType>{1.f, 1.f}(scale_b);
#endif
        ck_tile::FillUniformDistribution<ADataType>{0.f, 0.f}(a_host);
        // a_host.ForEach([](auto& self, auto idx) {
        //     float v = idx[0] % 3;
        //     // if(idx[1] % 4 == 0)
        //     self(idx) = ck_tile::type_convert<ADataType>(ck_tile::fp32x2_t{v, v});
        // });
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_host);

        ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_origin_host);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_origin_host);

        ck_tile::FillUniformDistribution<ScaleType>{1.f, 1.f}(scale_a);
        ck_tile::FillUniformDistribution<ScaleType>{1.f, 1.f}(scale_b);
    }
    else
    {
        throw std::runtime_error("wrong! Unexpected init_method");
    }

    ck_tile::HostTensor<BDataType> b_shuffled_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    preShuffleWeight<GemmConfig>(b_origin_host.begin(), b_shuffled_host.begin(), N, K);

    const auto scale_a_shuffled = preShuffleScale<GemmConfig, true>(scale_a);
    const auto scale_b_shuffled = preShuffleScale<GemmConfig, false>(scale_b);

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_shuffled_dev_buf(b_shuffled_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    b_shuffled_dev_buf.ToDevice(b_shuffled_host.data());
    c_rslt_host.SetZero();
    scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
    scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

    auto scale_a_dev_ptr = ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK>{
        static_cast<float*>(scale_a_dev_buf.GetDeviceBuffer()), M / ScaleGranularityM};
    auto scale_b_dev_ptr = ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK>{
        static_cast<float*>(scale_b_dev_buf.GetDeviceBuffer()), N / ScaleGranularityN};

    invoke_gemm_mx<GemmConfig,
                   ADataType,
                   BDataType,
                   ck_tile::tuple<>,
                   AccDataType,
                   CDataType,
                   ALayout,
                   BLayout,
                   ck_tile::tuple<>,
                   CLayout,
                   decltype(scale_a_dev_ptr),
                   decltype(scale_b_dev_ptr),
                   UsePersistentKernel>(a_dev_buf,
                                        b_shuffled_dev_buf,
                                        c_dev_buf,
                                        M,
                                        N,
                                        K,
                                        stride_A,
                                        stride_B,
                                        stride_C,
                                        kbatch,
                                        scale_a_dev_ptr,
                                        scale_b_dev_ptr,
                                        n_warmup,
                                        n_repeat);

    c_dev_buf.FromDevice(c_rslt_host.data());

    bool pass = true;
    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm_mx<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
            a_host, b_origin_host, c_m_n_host_ref, scale_a, scale_b);

        const float rtol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;
        const float atol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;

        pass = ck_tile::check_err(
            c_rslt_host, c_m_n_host_ref, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass ? 0 : -1;
}

template <typename GemmConfig>
int run_gemm_mx_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string mx_prec  = arg_parser.get_str("mx_prec");
    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");
    int persistent_opt   = arg_parser.get_int("persistent");

    if(a_layout == "R" && b_layout == "C")
    {
        if(mx_prec == "fp4xfp4")
        {
            if(persistent_opt == 0)
            {
                return run_gemm_mx_with_layouts<ck_tile::pk_fp4_t,
                                                ck_tile::pk_fp4_t,
                                                ck_tile::fp16_t,
                                                GemmConfig,
                                                false>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                throw std::runtime_error("Only support non-persistent kernel now!");
                // return run_gemm_mx_with_layouts<ck_tile::pk_fp4_t,
                //                                   ck_tile::pk_fp4_t,
                //                                   ck_tile::fp16_t,
                //                                   GemmConfig,
                //                                   true>(argc, argv, Row{}, Col{}, Row{});
            }
        }
        else if(mx_prec == "fp6xfp6")
        {
            throw std::runtime_error("Only support fp4xfp4 now!");
        }
        else if(mx_prec == "fp8xfp8")
        {
            throw std::runtime_error("Only support fp4xfp4 now!");
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
    return -1;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return EXIT_FAILURE;
    try
    {
        int warp_tile = arg_parser.get_int("warp_tile");
        if(warp_tile == 0)
        {
            return run_gemm_mx_example<MXfp4_GemmConfig16>(argc, argv);
        }
        else if(warp_tile == 1)
        {
            throw std::runtime_error("Only support MFMA_16x16x128 now!");
        }
        else
        {
            throw std::runtime_error("Unsupported warp_tile!");
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
