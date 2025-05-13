// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_mx.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"

namespace ck {
namespace profiler {

#if 1
template <bool KLast>
inline void preShuffleScaleBuffer(ck::e8m0_bexp_t* src, ck::e8m0_bexp_t* dst, int MN, int K)
{
    int MNXdlPack = 2;
    int KXdlPack  = 2;

    int XdlMNThread = 16;
    int XdlKThread  = 64 / XdlMNThread;

    int K0 = K / KXdlPack / XdlKThread; // KRepeat

    // The 4 16x128 building blocks will be packed into 1 32x256 for F4
    // The 8 16x16x128 mfma will be packed into 1 32x32x256 for F4

    // unfold the MN32xK(256/32) scale buffer
    //    4            16             2           2
    // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
    // Then, MNRepeat->KRepeat

    for(int n = 0; n < MN; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
            int tempn = n % (XdlMNThread * MNXdlPack);
            int n1    = tempn % XdlMNThread; // i XdlMNThread
            int n2    = tempn / XdlMNThread; // i MNXdlPack

            int k0    = k / (XdlKThread * KXdlPack); // i KRepeat
            int tempk = k % (XdlKThread * KXdlPack);
            int k1    = tempk % XdlKThread; // i XdlKThread
            int k2    = tempk / XdlKThread; // i KXdlPack

            int outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                              k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                              k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                              k2 * MNXdlPack + n2;
            // src[n * K + k] = ck::type_convert<ck::e8m0_bexp_t>(static_cast<float>(powf(2.0f, n2 +
            // k2 * MNXdlPack)));
            if constexpr(KLast)
                dst[outputIndex] = src[n * K + k];
            else
                dst[outputIndex] = src[k * MN + n];
        }
    }
}
#endif

template <typename ADataType,
          typename BDataType,
          typename XDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename ELayout>
bool profile_gemm_mx_impl(int do_verification,
                          int init_method,
                          bool do_log,
                          bool time_kernel,
                          int M,
                          int N,
                          int K,
                          int StrideA,
                          int StrideB,
                          int StrideC,
                          int n_warmup,
                          int n_iter,
                          uint64_t rotating = 0)
{
    constexpr int ScaleBlockSize = 32;
    bool pass                    = true;

    // Hardcode scale layouts as per pipeline assumptions
    // TODO: Allow user to specify scale layouts
    using AScaleLayout = tensor_layout::gemm::RowMajor;
    using BScaleLayout = tensor_layout::gemm::ColumnMajor;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };
    auto f_get_default_stride =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<ck::index_t>(col);
                }
                else
                {
                    return static_cast<ck::index_t>(row);
                }
            }
            else
                return static_cast<ck::index_t>(stride);
        };
    auto Scale_Stride_AM = f_get_default_stride(M, K / ScaleBlockSize, -1, AScaleLayout{});
    auto Scale_Stride_BN = f_get_default_stride(K / ScaleBlockSize, N, -1, BScaleLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    Tensor<XDataType> a_m_k_scale(f_host_tensor_descriptor(
        M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{})); // scales for A
    Tensor<XDataType> b_k_n_scale(f_host_tensor_descriptor(
        K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{})); // scales for B

    Tensor<XDataType> a_shuffled_scale(f_host_tensor_descriptor(
        M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{})); // scales for A
    Tensor<XDataType> b_shuffled_scale(f_host_tensor_descriptor(
        K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{})); // scales for B

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, ELayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, ELayout{}));

    int total_gemm_needed =
        a_m_k.GetElementSpaceSizeInBytes() + b_k_n.GetElementSpaceSizeInBytes() +
        a_m_k_scale.GetElementSpaceSizeInBytes() + b_k_n_scale.GetElementSpaceSizeInBytes();
    int rotating_count = std::max(
        1,
        std::min(n_iter,
                 static_cast<int>(std::ceil(static_cast<double>(rotating) / total_gemm_needed))));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "a_m_k_scale: " << a_m_k_scale.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "b_k_n_scale: " << b_k_n_scale.mDesc << std::endl;
    std::cout << "e_m_n: " << c_m_n_device_result.mDesc << std::endl;
    std::cout << "rotating count: " << rotating_count << std::endl;

    auto a_data_element = [](float x) {
        if constexpr(ck::is_same_v<ADataType, ck::f4x2_pk_t>)
            return ck::type_convert<ADataType>(ck::float2_t(x));
        else
            return ck::type_convert<ADataType>(x);
    };
    auto b_data_element = [](float x) {
        if constexpr(ck::is_same_v<BDataType, ck::f4x2_pk_t>)
            return ck::type_convert<BDataType>(ck::float2_t(x));
        else
            return ck::type_convert<BDataType>(x);
    };
    static_assert(ck::is_same_v<XDataType, ck::e8m0_bexp_t>);
    switch(init_method)
    {
    case 0: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{a_data_element(1.0f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(1.0f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{b_data_element(2.0f)}(b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(0.5f)}(b_k_n_scale);
        break;

    case 1:
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 6}); // Z[-5,5]
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 6}); // Z[-5,5]
        // scales: {0.25, 0.5, 1, 2}
        a_m_k_scale.GenerateTensorValue(GeneratorTensor_2<XDataType>{120, 129});
        b_k_n_scale.GenerateTensorValue(GeneratorTensor_2<XDataType>{125, 129});
        break;
    }

#if 1
    preShuffleScaleBuffer<ck::is_same_v<ALayout, tensor_layout::gemm::RowMajor>>(
        a_m_k_scale.mData.data(), a_shuffled_scale.mData.data(), M, K / ScaleBlockSize);
    preShuffleScaleBuffer<ck::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>>(
        b_k_n_scale.mData.data(), b_shuffled_scale.mData.data(), N, K / ScaleBlockSize);
#endif

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    using CElementOp = PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.GetElementSpaceSize());
    DeviceMem a_scale_device_buf(sizeof(XDataType) * a_m_k_scale.GetElementSpaceSize());
    DeviceMem b_scale_device_buf(sizeof(XDataType) * b_k_n_scale.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    a_scale_device_buf.ToDevice(a_m_k_scale.mData.data());
    b_scale_device_buf.ToDevice(b_k_n_scale.mData.data());

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMX<ALayout,
                                                                BLayout,
                                                                ELayout,
                                                                ADataType,
                                                                XDataType,
                                                                BDataType,
                                                                XDataType,
                                                                CDataType,
                                                                ScaleBlockSize,
                                                                AElementOp,
                                                                BElementOp,
                                                                CElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // Run reference GEMM
    if(do_verification)
    {
        Tensor<AccDataType> c_m_n({M, N});

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMXGemm< //
            ADataType,
            BDataType,
            CDataType,
            AccDataType,
            XDataType,
            PassThrough,
            PassThrough,
            PassThrough,
            float,
            float>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_m_k,
                                                  a_m_k_scale,
                                                  b_k_n,
                                                  b_k_n_scale,
                                                  c_m_n_host_result,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device GEMM instances
    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr =
            op_ptr->MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                        static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                        std::array<const void*, 0>{},
                                        static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                        M,
                                        N,
                                        K,
                                        StrideA,
                                        StrideB,
                                        std::array<ck::index_t, 0>{},
                                        StrideC,
                                        a_scale_device_buf.GetDeviceBuffer(),
                                        b_scale_device_buf.GetDeviceBuffer(),
                                        a_element_op,
                                        b_element_op,
                                        c_element_op);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {

            // re-init C to zero before profiling next kernel
            c_device_buf.SetZero();

            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false, 0, n_warmup, n_iter});

            if(do_verification)
            {
                c_device_buf.FromDevice(c_m_n_device_result.mData.data());
                pass = pass & ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                        << std::endl;
                }
            }

            std::string op_name = op_ptr->GetTypeString();

            float ave_time = invoker_ptr->Run(
                argument_ptr.get(),
                StreamConfig{
                    nullptr, time_kernel, 0, n_warmup, n_iter, rotating_count > 1, rotating_count});

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype =                                     //
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + //
                sizeof(CDataType) * M * N +                             //
                sizeof(XDataType) * (M * K + K * N) / ScaleBlockSize;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    }

    if constexpr(is_same<CDataType, float>::value)
    {
        std::cout << "Best Perf for datatype = f32";
    }
    else if constexpr(is_same<CDataType, half_t>::value)
    {
        std::cout << "Best Perf for datatype = f16";
    }
    else if constexpr(is_same<CDataType, bhalf_t>::value)
    {
        std::cout << "Best Perf for datatype = bf16";
    }
    else if constexpr(is_same<CDataType, int8_t>::value)
    {
        std::cout << "Best Perf for datatype = int8";
    }

    if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " ALayout =  RowMajor";
    }
    else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " ALayout =  ColumnMajor";
    }

    if constexpr(is_same<BLayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " BLayout =  RowMajor";
    }
    else if constexpr(is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " BLayout =  ColumnMajor";
    }

    std::cout << " M = " << M << " N = " << N << " K = " << K << " StrideA = " << StrideA
              << " StrideB = " << StrideB << " StrideC = " << StrideC << " : " << best_ave_time
              << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec << " GB/s, "
              << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
