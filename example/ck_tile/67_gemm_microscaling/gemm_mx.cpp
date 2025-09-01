// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"

#define EXAMPLE_A_LAYOUT RowMajor
#define EXAMPLE_B_LAYOUT ColMajor
#define EXAMPLE_C_LAYOUT RowMajor

#define EXAMPLE_A_DTYPE fp8
#define EXAMPLE_B_DTYPE fp8
#define EXAMPLE_C_DTYPE fp16

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)

using ALayout = ck_tile::tensor_layout::gemm::EXAMPLE_A_LAYOUT;
using BLayout = ck_tile::tensor_layout::gemm::EXAMPLE_B_LAYOUT;
using CLayout = ck_tile::tensor_layout::gemm::EXAMPLE_C_LAYOUT;

using ADataType   = ck_tile::CONCAT(EXAMPLE_A_DTYPE, _t);
using BDataType   = ck_tile::CONCAT(EXAMPLE_B_DTYPE, _t);
using CDataType   = ck_tile::CONCAT(EXAMPLE_C_DTYPE, _t);
using AccDataType = float;

template <typename Layout>
struct is_row_major
    : ck_tile::bool_constant<
          std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>
{
};

template <typename GemmConfig, bool Persistent, typename CDEElementWise>
float gemm(const ck_tile::GemmHostArgs</*NumDTensor = 0*/>& args, const ck_tile::stream_config& s)

{
    if constexpr(Persistent)
        std::cout << "WARNING: Ignoring persistent kernel option for basic gemm." << std::endl;
    // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 1;

    // This part comes from the Codegen
    constexpr ck_tile::index_t M_Tile = 256;
    constexpr ck_tile::index_t N_Tile = 256;
    constexpr ck_tile::index_t K_Tile = 64;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 16;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    using CodegenPipelineProblem = ck_tile::
        GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;

    using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto memory_operation = memory_operation_.value;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             CodegenPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << CodegenGemmShape::GetName() << '\n'
                      << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                      << "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    };

    if(args.k_batch == 1)
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::atomic_add>{});
    }
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename GemmConfig>
float gemm(const ck_tile::GemmHostArgs<>& args, const ck_tile::stream_config& s);

template <typename GemmConfig>
float invoke_gemm(ck_tile::HostTensor<ADataType>& a_m_k,
                  ck_tile::HostTensor<BDataType>& b_k_n,
                  ck_tile::HostTensor<CDataType>& c_m_n,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat)
{
    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n.get_element_space_size_in_bytes());
    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.ToDevice(c_m_n.data());

    ck_tile::GemmHostArgs</*NumDTensor = 0*/> args = {a_m_k_dev_buf.GetDeviceBuffer(),
                                                      b_k_n_dev_buf.GetDeviceBuffer(),
                                                      {},
                                                      c_m_n_dev_buf.GetDeviceBuffer(),
                                                      kbatch,
                                                      M,
                                                      N,
                                                      K,
                                                      stride_A,
                                                      stride_B,
                                                      {},
                                                      stride_C};

    float ave_time;
    {
        ave_time = gemm<GemmConfig,
                        ADataType,
                        BDataType,
                        DsDataType,
                        AccDataType,
                        CDataType,
                        ALayout,
                        BLayout,
                        DsLayout,
                        CLayout,
                        false,
                        CDEElementWise>(
            args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});
    }

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm MX kernel with M=" << M << " N=" << N << " K=" << K
              << " StrideA=" << stride_A << " StrideB=" << stride_B << " StrideC=" << stride_C
              << " A_Layout=" << ALayout::name << " B_Layout =" << BLayout::name
              << " C_Layout=" << CLayout::name << " A_Type=" << DataTypeTraits<ADataType>::name
              << " B_Type=" << DataTypeTraits<BDataType>::name
              << " C_Type=" << DataTypeTraits<CDataType>::name //
              << ": " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    return ave_time;
}

template <typename GemmConfig>
int run_gemm_mx_example(int argc, char* argv[])
{
    const ExampleArgs args(argc, argv);

    const auto stride_A = ck_tile::get_default_stride(args.M, args.K, 0, is_row_major<ALayout>{});
    const auto stride_B = ck_tile::get_default_stride(args.K, args.N, 0, is_row_major<BLayout>{});
    const auto stride_C = ck_tile::get_default_stride(args.M, args.N, 0, is_row_major<CLayout>{});

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, 0, is_row_major<ALayout>{}));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, 0, is_row_major<BLayout>{}));
    ck_tile::HostTensor<CDataType> c_m_n(
        ck_tile::host_tensor_descriptor(M, N, 0, is_row_major<CLayout>{}));
    ck_tile::HostTensor<CDataType> c_m_n_ref(
        ck_tile::host_tensor_descriptor(M, N, 0, is_row_major<CLayout>{}));

    if(args.init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
    }
    else if(args.init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
    }
    else if(args.init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
    }
    c_m_n.SetValue(-ck_tile::numeric<VDataType>::max());
    c_m_n_ref.SetZero();

    invoke_gemm<GemmConfig>(a_m_k,
                            b_k_n,
                            c_m_n,
                            M,
                            N,
                            K,
                            stride_A,
                            stride_B,
                            stride_C,
                            args.n_warmup,
                            args.n_repeat);

    c_m_n_dev_buf.FromDevice(c_m_n.data());
    bool pass = true;

    if(args.validate == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    } else if (args.validate != 0) {
        std::cout << "Wrong! validate flag only can be 0(no validation) or 1(with validation)!"
                  << std::endl;
        pass = false;
    }
    return pass;
}

int main(int argc, char* argv[])
{
    try
    {
        return !run_gemm_mx_example<GemmMXConfigBase>(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
