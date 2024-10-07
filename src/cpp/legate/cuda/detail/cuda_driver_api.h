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
#include <string>
#include <string_view>

#if defined(_WIN64) || defined(__LP64__)
// Don't use uint64_t, we want to match the driver headers exactly
using CUdeviceptr = unsigned long long;  // NOLINT(google-runtime-int)
#else
using CUdeviceptr = unsigned int;
#endif
static_assert(sizeof(CUdeviceptr) == sizeof(void*));

using CUresult  = int;
using CUdevice  = int;
using CUcontext = struct CUctx_st*;
using CUstream  = struct CUstream_st*;

namespace legate::cuda::detail {

class CUDADriverAPI {
 public:
  CUDADriverAPI();

  [[nodiscard]] CUresult init(unsigned int flags) const;
  [[nodiscard]] CUresult stream_create(CUstream* stream, unsigned int flags) const;
  [[nodiscard]] CUresult get_error_string(CUresult error, const char** str) const;
  [[nodiscard]] CUresult get_error_name(CUresult error, const char** str) const;
  [[nodiscard]] CUresult pointer_get_attributes(void* data, int attribute, CUdeviceptr ptr) const;
  [[nodiscard]] CUresult mem_cpy_async(CUdeviceptr dst,
                                       CUdeviceptr src,
                                       std::size_t num_bytes,
                                       CUstream stream) const;
  [[nodiscard]] CUresult mem_cpy(CUdeviceptr dst, CUdeviceptr src, std::size_t num_bytes) const;
  [[nodiscard]] CUresult stream_destroy(CUstream stream) const;
  [[nodiscard]] CUresult stream_synchronize(CUstream stream) const;
  [[nodiscard]] CUresult ctx_synchronize() const;

  [[nodiscard]] std::string_view handle_path() const noexcept;
  [[nodiscard]] bool is_loaded() const noexcept;

 private:
  std::string handle_path_{};
  void* handle_{};

  void read_symbols_();
  void check_initialized_() const;

  CUresult (*init_)(unsigned int flags)                                           = nullptr;
  CUresult (*stream_create_)(CUstream* stream, unsigned int flags)                = nullptr;
  CUresult (*get_error_string_)(CUresult error, const char** str)                 = nullptr;
  CUresult (*get_error_name_)(CUresult error, const char** str)                   = nullptr;
  CUresult (*pointer_get_attributes_)(void* data, int attribute, CUdeviceptr ptr) = nullptr;
  CUresult (*mem_cpy_async_)(CUdeviceptr dst,
                             CUdeviceptr src,
                             std::size_t num_bytes,
                             CUstream stream)                                     = nullptr;
  CUresult (*mem_cpy_)(CUdeviceptr dst, CUdeviceptr src, std::size_t num_bytes)   = nullptr;
  CUresult (*stream_destroy_)(CUstream stream)                                    = nullptr;
  CUresult (*stream_synchronize_)(CUstream stream)                                = nullptr;
  CUresult (*ctx_synchronize_)()                                                  = nullptr;
};

}  // namespace legate::cuda::detail
