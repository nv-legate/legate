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

#include <legate/cuda/detail/module_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <array>
#include <fmt/format.h>
#include <stdexcept>

namespace legate::cuda::detail {

const std::unordered_map<const void*, CUlibrary>& CUDAModuleManager::libraries_() const noexcept
{
  return libs_;
}

std::unordered_map<const void*, CUlibrary>& CUDAModuleManager::libraries_() noexcept
{
  return libs_;
}

// ==========================================================================================

CUDAModuleManager::CUDAModuleManager(InternalSharedPtr<CUDADriverAPI> driver_api)
  : driver_api_{std::move(driver_api)}
{
}

CUDAModuleManager::~CUDAModuleManager() noexcept
{
  for (auto&& [_, cu_lib] : libraries_()) {
    driver_api_->library_unload(&cu_lib);
  }
}

// See
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d
#define CU_JIT_GENERATE_DEBUG_INFO ((CUjit_option)11)
#define CU_JIT_GENERATE_LINE_INFO ((CUjit_option)13)

/*static*/ const std::pair<Span<CUjit_option>, Span<void*>>&
CUDAModuleManager::default_jit_options()
{
  static constexpr int one  = 1;
  static constexpr int zero = 0;

  static constexpr auto* one_v  = const_cast<void*>(static_cast<const void*>(&one));
  static constexpr auto* zero_v = const_cast<void*>(static_cast<const void*>(&zero));

  static std::array keys   = {CU_JIT_GENERATE_LINE_INFO, CU_JIT_GENERATE_DEBUG_INFO};
  static std::array values = {LEGATE_DEFINED(LEGATE_USE_DEBUG) ? one_v : zero_v,
                              LEGATE_DEFINED(LEGATE_USE_DEBUG) ? one_v : zero_v};

  static constexpr std::pair<Span<CUjit_option>, Span<void*>> options = {
    {keys.begin(), keys.end()}, {values.begin(), values.end()}};

  return options;
}

/*static*/ const std::pair<Span<CUlibraryOption>, Span<void*>>&
CUDAModuleManager::default_library_options()
{
  static constexpr std::pair<Span<CUlibraryOption>, Span<void*>> options{};

  return options;
}

CUlibrary CUDAModuleManager::load_library(
  const void* fatbin,
  std::pair<Span<CUjit_option>, Span<void*>> jit_options,
  std::pair<Span<CUlibraryOption>, Span<void*>> library_options)
{
  if (!fatbin) {
    throw legate::detail::TracedException<std::invalid_argument>{"Fatbin pointer cannot be NULL"};
  }

  if (jit_options.first.size() != jit_options.second.size()) {
    throw legate::detail::TracedException<std::out_of_range>{
      fmt::format("Number of jit options ({}) != number of jit option values ({})",
                  jit_options.first.size(),
                  jit_options.second.size())};
  }
  if (library_options.first.size() != library_options.second.size()) {
    throw legate::detail::TracedException<std::out_of_range>{
      fmt::format("Number of library options ({}) != number of library option values ({})",
                  library_options.first.size(),
                  library_options.second.size())};
  }

  // Need to acquire and hold the lock here because this can be called from anywhere, including
  // within tasks. In that case, we don't want the initialization of the module to introduce a
  // race condition.
  const std::lock_guard<std::mutex> lock{mut_};

  const auto [it, inserted] = libraries_().try_emplace(fatbin);

  if (inserted) {
    try {
      const auto* api = legate::detail::Runtime::get_runtime()->get_cuda_driver_api();

      it->second = api->library_load_data(fatbin,
                                          jit_options.first.begin(),
                                          jit_options.second.begin(),
                                          jit_options.first.size(),
                                          library_options.first.begin(),
                                          library_options.second.begin(),
                                          library_options.first.size());
    } catch (...) {
      libraries_().erase(it);
      throw;
    }
  }
  return it->second;
}

CUkernel CUDAModuleManager::load_kernel_from_fatbin(const void* fatbin, const char* kernel_name)
{
  // No need to check fatbin, load_library() does that for us
  if (!kernel_name) {
    throw legate::detail::TracedException<std::invalid_argument>{"Kernel name must not be NULL"};
  }

  const auto lib = load_library(fatbin);

  return legate::detail::Runtime::get_runtime()->get_cuda_driver_api()->library_get_kernel(
    lib, kernel_name);
}

}  // namespace legate::cuda::detail
