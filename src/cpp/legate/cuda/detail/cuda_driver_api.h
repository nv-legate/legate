/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

// NOLINTBEGIN
#if defined(_WIN64) || defined(__LP64__)
// Don't use uint64_t, we want to match the driver headers exactly
using CUdeviceptr = unsigned long long;  // NOLINT(google-runtime-int)
#else
using CUdeviceptr = unsigned int;
#endif
static_assert(sizeof(CUdeviceptr) == sizeof(void*));

using CUresult   = int;
using CUdevice   = int;
using CUcontext  = struct CUctx_st*;
using CUstream   = struct CUstream_st*;
using CUevent    = struct CUevent_st*;
using CUfunction = struct CUfunc_st*;
using CUlibrary  = struct CUlib_st*;
using CUkernel   = struct CUkern_st*;

enum CUlibraryOption : int;
enum CUjit_option : int;

#undef CU_STREAM_NON_BLOCKING
#define CU_STREAM_NON_BLOCKING 0x1
// NOLINTEND

namespace legate::cuda::detail {

class Dim3 {
 public:
  constexpr Dim3(std::size_t x_ = 1,  // NOLINT(google-explicit-constructor)
                 std::size_t y_ = 1,
                 std::size_t z_ = 1) noexcept
    : x{x_}, y{y_}, z{z_}
  {
  }

  std::size_t x{1};
  std::size_t y{1};
  std::size_t z{1};
};

class CUDADriverAPI {
 public:
  CUDADriverAPI();

  [[nodiscard]] CUresult init(unsigned int flags) const;

  [[nodiscard]] CUresult get_error_string(CUresult error, const char** str) const;
  [[nodiscard]] CUresult get_error_name(CUresult error, const char** str) const;

  [[nodiscard]] CUresult pointer_get_attributes(void* data, int attribute, CUdeviceptr ptr) const;

  [[nodiscard]] CUresult mem_cpy_async(CUdeviceptr dst,
                                       CUdeviceptr src,
                                       std::size_t num_bytes,
                                       CUstream stream) const;
  template <typename T, typename U>
  [[nodiscard]] CUresult mem_cpy_async(T* dst,
                                       const U* src,
                                       std::size_t num_bytes,
                                       CUstream stream) const;
  [[nodiscard]] CUresult mem_cpy(CUdeviceptr dst, CUdeviceptr src, std::size_t num_bytes) const;
  template <typename T, typename U>
  [[nodiscard]] CUresult mem_cpy(T* dst, const U* src, std::size_t num_bytes) const;

  [[nodiscard]] CUresult stream_create(CUstream* stream, unsigned int flags) const;
  [[nodiscard]] CUresult stream_destroy(CUstream stream) const;
  [[nodiscard]] CUresult stream_synchronize(CUstream stream) const;

  [[nodiscard]] CUresult event_create(CUevent* event, unsigned int flags) const;
  [[nodiscard]] CUresult event_record(CUevent event, CUstream stream) const;
  [[nodiscard]] CUresult event_synchronize(CUevent event) const;
  [[nodiscard]] CUresult event_elapsed_time(float* ms, CUevent start, CUevent end) const;
  [[nodiscard]] CUresult event_destroy(CUevent event) const;

  [[nodiscard]] CUresult ctx_get_device(CUdevice* device) const;
  [[nodiscard]] CUresult ctx_synchronize() const;

  [[nodiscard]] CUresult launch_kernel(CUfunction f,
                                       Dim3 grid_dim,
                                       Dim3 block_dim,
                                       std::size_t shared_mem_bytes,
                                       CUstream stream,
                                       void** kernel_params,
                                       void** extra) const;
  [[nodiscard]] CUresult launch_kernel(CUkernel f,
                                       Dim3 grid_dim,
                                       Dim3 block_dim,
                                       std::size_t shared_mem_bytes,
                                       CUstream stream,
                                       void** kernel_params,
                                       void** extra) const;

  [[nodiscard]] CUresult library_load_data(CUlibrary* library,
                                           const void* code,
                                           CUjit_option* jit_options,
                                           void** jit_options_values,
                                           std::size_t num_jit_options,
                                           CUlibraryOption* library_options,
                                           void** library_option_values,
                                           std::size_t num_library_options) const;
  [[nodiscard]] CUresult library_get_kernel(CUkernel* kernel,
                                            CUlibrary library,
                                            const char* name) const;
  [[nodiscard]] CUresult library_unload(CUlibrary library) const;

  [[nodiscard]] std::string_view handle_path() const noexcept;
  [[nodiscard]] bool is_loaded() const noexcept;

 private:
  std::string handle_path_{};
  void* handle_{};

  void read_symbols_();
  void check_initialized_() const;

  CUresult (*init_)(unsigned int flags) = nullptr;

  CUresult (*get_error_string_)(CUresult error, const char** str) = nullptr;
  CUresult (*get_error_name_)(CUresult error, const char** str)   = nullptr;

  CUresult (*pointer_get_attributes_)(void* data, int attribute, CUdeviceptr ptr) = nullptr;

  CUresult (*mem_cpy_async_)(CUdeviceptr dst,
                             CUdeviceptr src,
                             std::size_t num_bytes,
                             CUstream stream)                                   = nullptr;
  CUresult (*mem_cpy_)(CUdeviceptr dst, CUdeviceptr src, std::size_t num_bytes) = nullptr;

  CUresult (*stream_create_)(CUstream* stream, unsigned int flags) = nullptr;
  CUresult (*stream_destroy_)(CUstream stream)                     = nullptr;
  CUresult (*stream_synchronize_)(CUstream stream)                 = nullptr;

  CUresult (*event_create_)(CUevent* event, unsigned int flags)          = nullptr;
  CUresult (*event_record_)(CUevent event, CUstream stream)              = nullptr;
  CUresult (*event_synchronize_)(CUevent event)                          = nullptr;
  CUresult (*event_elapsed_time_)(float* ms, CUevent start, CUevent end) = nullptr;
  CUresult (*event_destroy_)(CUevent event)                              = nullptr;

  CUresult (*ctx_get_device_)(CUdevice* device) = nullptr;
  CUresult (*ctx_synchronize_)()                = nullptr;

  CUresult (*launch_kernel_)(CUfunction f,
                             unsigned int grid_dim_x,
                             unsigned int grid_dim_y,
                             unsigned int grid_dim_z,
                             unsigned int block_dim_x,
                             unsigned int block_dim_y,
                             unsigned int block_dim_z,
                             unsigned int shared_mem_bytes,
                             CUstream stream,
                             void** kernel_params,
                             void** extra) = nullptr;

  CUresult (*library_load_data_)(CUlibrary* library,
                                 const void* code,
                                 CUjit_option* jit_options,
                                 void** jit_options_values,
                                 std::size_t num_jit_options,
                                 CUlibraryOption* library_options,
                                 void** library_option_values,
                                 std::size_t num_library_options)                        = nullptr;
  CUresult (*library_get_kernel_)(CUkernel* kernel, CUlibrary library, const char* name) = nullptr;
  CUresult (*library_unload_)(CUlibrary library)                                         = nullptr;
};

class CUDADriverError : public std::runtime_error {
 public:
  CUDADriverError(const std::string& what, CUresult result);

  [[nodiscard]] CUresult error_code() const noexcept;

 private:
  CUresult result_{};
};

[[noreturn]] void throw_cuda_driver_error(CUresult result,
                                          std::string_view expression,
                                          std::string_view file,
                                          std::string_view func,
                                          int line);

#define LEGATE_CHECK_CUDRIVER(...)                                                          \
  do {                                                                                      \
    const ::CUresult __legate_cu_result__ = __VA_ARGS__;                                    \
    if (LEGATE_UNLIKELY(__legate_cu_result__)) {                                            \
      ::legate::cuda::detail::throw_cuda_driver_error(                                      \
        __legate_cu_result__, LEGATE_STRINGIZE(__VA_ARGS__), __FILE__, __func__, __LINE__); \
    }                                                                                       \
  } while (0)

}  // namespace legate::cuda::detail

#include "legate/cuda/detail/cuda_driver_api.inl"
