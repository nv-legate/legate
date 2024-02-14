/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/task/task_info.h"
#include "core/task/variant_options.h"

#include "legion.h"

#include <optional>
#include <string_view>

namespace legate::detail {

void task_wrapper(VariantImpl,
                  LegateVariantCode,
                  std::optional<std::string_view>,
                  const void*,
                  size_t,
                  const void*,
                  size_t,
                  Legion::Processor);

template <VariantImpl variant_fn, LegateVariantCode variant_kind>
inline void task_wrapper_dyn_name(const void* args,
                                  std::size_t arglen,
                                  const void* userdata,
                                  std::size_t userlen,
                                  Legion::Processor p)
{
  task_wrapper(variant_fn, variant_kind, {}, args, arglen, userdata, userlen, std::move(p));
}

template <typename T>
using void_t = void;

template <typename T, typename = void>
struct CPUVariant : std::false_type {};

template <typename T, typename = void>
struct OMPVariant : std::false_type {};

template <typename T, typename = void>
struct GPUVariant : std::false_type {};

template <typename T>
struct CPUVariant<T, void_t<decltype(T::cpu_variant)>> : std::true_type {
  static constexpr auto variant = T::cpu_variant;
  static constexpr auto id      = LEGATE_CPU_VARIANT;
};

template <typename T>
struct OMPVariant<T, void_t<decltype(T::omp_variant)>> : std::true_type {
  static constexpr auto variant = T::omp_variant;
  static constexpr auto id      = LEGATE_OMP_VARIANT;
};

template <typename T>
struct GPUVariant<T, void_t<decltype(T::gpu_variant)>> : std::true_type {
  static constexpr auto variant = T::gpu_variant;
  static constexpr auto id      = LEGATE_GPU_VARIANT;
};

template <typename T, template <typename...> typename SELECTOR, bool VALID = SELECTOR<T>::value>
struct VariantHelper {
  static void record(TaskInfo* /*task_info*/,
                     const std::map<LegateVariantCode, VariantOptions>& /*all_options*/)
  {
  }
};

template <typename T, template <typename...> typename SELECTOR>
struct VariantHelper<T, SELECTOR, true> {
  static void record(TaskInfo* task_info,
                     const std::map<LegateVariantCode, VariantOptions>& all_options)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    constexpr auto variant_impl = SELECTOR<T>::variant;
    constexpr auto variant_kind = SELECTOR<T>::id;
    constexpr auto entry        = T::BASE::template task_wrapper_<variant_impl, variant_kind>;

    task_info->add_variant(variant_kind, variant_impl, entry, all_options);
  }
};

}  // namespace legate::detail
