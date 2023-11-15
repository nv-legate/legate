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

namespace legate::detail {

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
  static void record(TaskInfo* task_info,
                     const std::map<LegateVariantCode, VariantOptions>& all_options)
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
    constexpr auto VARIANT_IMPL = SELECTOR<T>::variant;
    constexpr auto WRAPPER      = T::BASE::template legate_task_wrapper<VARIANT_IMPL>;
    constexpr auto VARIANT_ID   = SELECTOR<T>::id;
    auto finder                 = all_options.find(VARIANT_ID);
    task_info->add_variant(VARIANT_ID,
                           VARIANT_IMPL,
                           Legion::CodeDescriptor(WRAPPER),
                           finder != all_options.end() ? finder->second : VariantOptions{});
  }
};

}  // namespace legate::detail
