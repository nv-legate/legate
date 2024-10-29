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

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/utilities/span.h>

#include <mutex>
#include <unordered_map>
#include <utility>

namespace legate::cuda::detail {

class CUDAModuleManager {
 public:
  ~CUDAModuleManager() noexcept;

  [[nodiscard]] static const std::pair<Span<CUjit_option>, Span<void*>>& default_jit_options();
  [[nodiscard]] static const std::pair<Span<CUlibraryOption>, Span<void*>>&
  default_library_options();

  [[nodiscard]] CUlibrary load_library(
    const void* fatbin,
    std::pair<Span<CUjit_option>, Span<void*>> jit_options        = default_jit_options(),
    std::pair<Span<CUlibraryOption>, Span<void*>> library_options = default_library_options());

  [[nodiscard]] CUkernel load_kernel_from_fatbin(const void* fatbin, const char* kernel_name);

 private:
  [[nodiscard]] const std::unordered_map<const void*, CUlibrary>& libraries_() const noexcept;
  [[nodiscard]] std::unordered_map<const void*, CUlibrary>& libraries_() noexcept;

  std::mutex mut_{};
  std::unordered_map<const void*, CUlibrary> libs_{};
};

}  // namespace legate::cuda::detail
