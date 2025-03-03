/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <mutex>
#include <unordered_map>
#include <utility>

namespace legate::cuda::detail {

class CUDAModuleManager {
 public:
  explicit CUDAModuleManager(InternalSharedPtr<CUDADriverAPI> driver_api);
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
  // Holding a reference to the CUDADriverAPI object here to keep it alive, instead of accessing it
  // through the runtime, for the following reason: when the program aborts before legate::finish is
  // invoked, the destructor of CUDAModuleManager is called as part of tearing down the singleton
  // Runtime object, and inside the destructor, calling Runtime::get_runtime would try to
  // reinitialized the Runtime, as RuntimeManager::rt_ had already been reset by the destructor.
  InternalSharedPtr<CUDADriverAPI> driver_api_{};
};

}  // namespace legate::cuda::detail
