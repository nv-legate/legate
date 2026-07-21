/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/task_info.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/type_traits.h>

#include <map>
#include <type_traits>

namespace legate {

class TaskContext;
class Library;

}  // namespace legate

namespace legate::detail {

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
    [[nodiscard]] static constexpr const VariantOptions* get_default_options_() noexcept     \
    {                                                                                        \
      if constexpr (detail::is_detected_v<has_default_variant_options, T>) {                 \
        static_assert(                                                                       \
          std::is_same_v<std::decay_t<decltype(T::NAME##_VARIANT_OPTIONS)>, VariantOptions>, \
          "Default variant options for " #NAME                                               \
          " variant has incompatible type. Expected static constexpr VariantOptions " #NAME  \
          "_VARIANT_OPTIONS = ...");                                                         \
        return &T::NAME##_VARIANT_OPTIONS;                                                   \
      } else {                                                                               \
        return nullptr;                                                                      \
      }                                                                                      \
    }                                                                                        \
                                                                                             \
   public:                                                                                   \
    static constexpr auto variant  = T::name##_variant;                                      \
    static constexpr auto id       = VariantCode::NAME;                                      \
    static constexpr auto* options = get_default_options_();                                 \
                                                                                             \
    static_assert(std::is_convertible_v<decltype(variant), VariantImpl>,                     \
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
                     const std::map<VariantCode, VariantOptions>& /*all_options*/,
                     legate::TaskInfo* /*task_info*/)
  {
  }
};

template <typename T, template <typename...> typename SELECTOR>
class VariantHelper<T, SELECTOR, true> {
 public:
  static void record(const legate::Library& lib,
                     const std::map<VariantCode, VariantOptions>& all_options,
                     legate::TaskInfo* task_info)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    constexpr auto variant_impl = SELECTOR<T>::variant;
    constexpr auto variant_kind = SELECTOR<T>::id;
    constexpr auto* options     = SELECTOR<T>::options;

    task_info->add_variant_(
      legate::TaskInfo::AddVariantKey{}, lib, variant_kind, variant_impl, options, all_options);
  }
};

}  // namespace legate::detail
