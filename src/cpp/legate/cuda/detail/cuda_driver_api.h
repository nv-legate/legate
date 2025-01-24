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

#include <legate/cuda/detail/cuda_driver_types.h>
#include <legate/utilities/macros.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

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

  void init() const;

  [[nodiscard]] const char* get_error_string(CUresult error) const;
  [[nodiscard]] const char* get_error_name(CUresult error) const;

  void mem_cpy_async(CUdeviceptr dst,
                     CUdeviceptr src,
                     std::size_t num_bytes,
                     CUstream stream) const;
  template <typename T, typename U>
  void mem_cpy_async(T* dst, const U* src, std::size_t num_bytes, CUstream stream) const;

  [[nodiscard]] CUstream stream_create(unsigned int flags) const;
  void stream_destroy(CUstream* stream) const;
  void stream_synchronize(CUstream stream) const;

  [[nodiscard]] CUevent event_create(unsigned int flags = 0) const;
  void event_record(CUevent event, CUstream stream) const;
  void event_synchronize(CUevent event) const;
  [[nodiscard]] float event_elapsed_time(CUevent start, CUevent end) const;
  void event_destroy(CUevent* event) const;

  void device_primary_ctx_retain(CUcontext* ctx, CUdevice dev) const;
  void device_primary_ctx_release(CUdevice dev) const;

  [[nodiscard]] CUdevice ctx_get_device() const;
  void ctx_push_current(CUcontext ctx) const;
  [[nodiscard]] CUcontext ctx_pop_current() const;
  void ctx_synchronize() const;

  [[nodiscard]] CUfunction kernel_get_function(CUkernel kernel) const;

  void launch_kernel_direct(CUfunction f,
                            Dim3 grid_dim,
                            Dim3 block_dim,
                            std::size_t shared_mem_bytes,
                            CUstream stream,
                            void** kernel_params,
                            void** extra) const;
  template <typename... T>
  void launch_kernel(CUkernel f,
                     Dim3 grid_dim,
                     Dim3 block_dim,
                     std::size_t shared_mem_bytes,
                     CUstream stream,
                     T&&... args) const;
  template <typename... T>
  void launch_kernel(CUfunction f,
                     Dim3 grid_dim,
                     Dim3 block_dim,
                     std::size_t shared_mem_bytes,
                     CUstream stream,
                     T&&... args) const;

  [[nodiscard]] CUlibrary library_load_data(const void* code,
                                            CUjit_option* jit_options,
                                            void** jit_options_values,
                                            std::size_t num_jit_options,
                                            CUlibraryOption* library_options,
                                            void** library_option_values,
                                            std::size_t num_library_options) const;
  [[nodiscard]] CUkernel library_get_kernel(CUlibrary library, const char* name) const;
  void library_unload(CUlibrary* library) const;

  [[nodiscard]] std::string_view handle_path() const noexcept;
  [[nodiscard]] bool is_loaded() const noexcept;

 private:
  std::string handle_path_{};
  void* handle_{};

  void read_symbols_();
  void check_initialized_() const;

  CUresult (*get_proc_address_)(const char* symbol,
                                void** pfn,
                                int cuda_version,
                                std::uint64_t flags) = nullptr;

  CUresult (*init_)(unsigned int flags) = nullptr;

  CUresult (*get_error_string_)(CUresult error, const char** str) = nullptr;
  CUresult (*get_error_name_)(CUresult error, const char** str)   = nullptr;

  CUresult (*mem_cpy_async_)(CUdeviceptr dst,
                             CUdeviceptr src,
                             std::size_t num_bytes,
                             CUstream stream) = nullptr;

  CUresult (*stream_create_)(CUstream* stream, unsigned int flags) = nullptr;
  CUresult (*stream_destroy_)(CUstream stream)                     = nullptr;
  CUresult (*stream_synchronize_)(CUstream stream)                 = nullptr;

  CUresult (*event_create_)(CUevent* event, unsigned int flags)          = nullptr;
  CUresult (*event_record_)(CUevent event, CUstream stream)              = nullptr;
  CUresult (*event_synchronize_)(CUevent event)                          = nullptr;
  CUresult (*event_elapsed_time_)(float* ms, CUevent start, CUevent end) = nullptr;
  CUresult (*event_destroy_)(CUevent event)                              = nullptr;

  CUresult (*device_primary_ctx_retain_)(CUcontext* ctx, CUdevice dev) = nullptr;
  CUresult (*device_primary_ctx_release_)(CUdevice dev)                = nullptr;

  CUresult (*ctx_get_device_)(CUdevice* device) = nullptr;
  CUresult (*ctx_push_current_)(CUcontext ctx)  = nullptr;
  CUresult (*ctx_pop_current_)(CUcontext* ctx)  = nullptr;
  CUresult (*ctx_synchronize_)()                = nullptr;

  CUresult (*kernel_get_function_)(CUfunction* func, CUkernel kernel) = nullptr;

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
                                 unsigned int num_jit_options,
                                 CUlibraryOption* library_options,
                                 void** library_option_values,
                                 unsigned int num_library_options)                       = nullptr;
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

}  // namespace legate::cuda::detail

#include <legate/cuda/detail/cuda_driver_api.inl>
