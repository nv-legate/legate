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

#include "core/task/task_info.h"
#include "core/task/variant_options.h"
#include "core/utilities/detail/type_traits.h"

#include "legion.h"

#include <optional>
#include <string_view>
#include <type_traits>

namespace legate {

class TaskContext;
class Library;

}  // namespace legate

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

#define LEGATE_SELECTOR_SPECIALIZATION(NAME, name)                                           \
  template <typename T, typename = void>                                                     \
  class NAME##Variant : public std::false_type {};                                           \
                                                                                             \
  template <typename T>                                                                      \
  class NAME##Variant<T, std::void_t<decltype(T::name##_variant)>> : public std::true_type { \
    /* Do not be fooled, U = T in all cases, but we need this to be a */                     \
    /* template for traits::is_detected. */                                                  \
    template <typename U>                                                                    \
    using has_default_variant_options = decltype(U::NAME##_VARIANT_OPTIONS);                 \
                                                                                             \
    [[nodiscard]] static constexpr const VariantOptions& get_default_options_() noexcept     \
    {                                                                                        \
      if constexpr (traits::detail::is_detected_v<has_default_variant_options, T>) {         \
        static_assert(                                                                       \
          std::is_same_v<std::decay_t<decltype(T::NAME##_VARIANT_OPTIONS)>, VariantOptions>, \
          "Default variant options for " #NAME                                               \
          " variant has incompatible type. Expected static constexpr VariantOptions " #NAME  \
          "_VARIANT_OPTIONS = ...");                                                         \
        return T::NAME##_VARIANT_OPTIONS;                                                    \
      } else {                                                                               \
        return VariantOptions::DEFAULT_OPTIONS;                                              \
      }                                                                                      \
    }                                                                                        \
                                                                                             \
   public:                                                                                   \
    static constexpr auto variant  = T::name##_variant;                                      \
    static constexpr auto id       = LEGATE_##NAME##_VARIANT;                                \
    static constexpr auto& options = get_default_options_();                                 \
                                                                                             \
    static_assert(std::is_convertible_v<decltype(variant), void (*)(legate::TaskContext)>,   \
                  "Malformed " #NAME                                                         \
                  " variant function. Variant function must have the following signature: "  \
                  "static void " #name "_variant(legate::TaskContext)");                     \
  }

LEGATE_SELECTOR_SPECIALIZATION(CPU, cpu);
LEGATE_SELECTOR_SPECIALIZATION(OMP, omp);
LEGATE_SELECTOR_SPECIALIZATION(GPU, gpu);

#undef LEGATE_SELECTOR_SPECIALIZATION

template <typename T, template <typename...> typename SELECTOR, bool VALID = SELECTOR<T>::value>
class VariantHelper {
 public:
  static void record(const legate::Library& /*lib*/,
                     TaskInfo* /*task_info*/,
                     const std::map<LegateVariantCode, VariantOptions>& /*all_options*/)
  {
  }
};

template <typename T, template <typename...> typename SELECTOR>
class VariantHelper<T, SELECTOR, true> {
 public:
  static void record(const legate::Library& lib,
                     TaskInfo* task_info,
                     const std::map<LegateVariantCode, VariantOptions>& all_options)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    constexpr auto variant_impl = SELECTOR<T>::variant;
    constexpr auto variant_kind = SELECTOR<T>::id;
    constexpr auto& options     = SELECTOR<T>::options;
    constexpr auto entry        = T::BASE::template task_wrapper_<variant_impl, variant_kind>;

    task_info->add_variant_(
      TaskInfo::AddVariantKey{}, lib, variant_kind, variant_impl, entry, options, all_options);
  }
};

}  // namespace legate::detail
