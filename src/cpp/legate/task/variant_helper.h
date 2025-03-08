/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/task_info.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/type_traits.h>

#include <legion.h>

#include <optional>
#include <string_view>
#include <type_traits>

namespace legate {

class TaskContext;
class Library;
class TaskConfig;

}  // namespace legate

namespace legate::detail {

template <typename T>
class LegionTask;

void task_wrapper(VariantImpl,
                  VariantCode,
                  std::optional<std::string_view>,
                  const void*,
                  std::size_t,
                  const void*,
                  std::size_t,
                  Legion::Processor);

template <VariantImpl variant_fn, VariantCode variant_kind>
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
    static_assert(std::is_convertible_v<decltype(variant), VariantImpl> ||                   \
                    std::is_same_v<typename T::BASE, LegionTask<T>>,                         \
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
                     const TaskConfig& /*task_config*/,
                     const std::map<VariantCode, VariantOptions>& /*all_options*/,
                     legate::TaskInfo* /*task_info*/)
  {
  }
};

template <typename T, template <typename...> typename SELECTOR>
class VariantHelper<T, SELECTOR, true> {
 public:
  static void record(const legate::Library& lib,
                     const TaskConfig& task_config,
                     const std::map<VariantCode, VariantOptions>& all_options,
                     legate::TaskInfo* task_info)
  {
    // Construct the code descriptor for this task so that the library
    // can register it later when it is ready
    constexpr auto variant_impl = SELECTOR<T>::variant;
    constexpr auto variant_kind = SELECTOR<T>::id;
    constexpr auto* options     = SELECTOR<T>::options;

    if constexpr (std::is_convertible_v<decltype(variant_impl), VariantImpl>) {
      constexpr auto entry = T::BASE::template task_wrapper_<variant_impl, variant_kind>;

      task_info->add_variant_(legate::TaskInfo::AddVariantKey{},
                              lib,
                              variant_kind,
                              variant_impl,
                              entry,
                              task_config,
                              options,
                              all_options);
    } else {
      using RET            = std::invoke_result_t<decltype(variant_impl),
                                                  const Legion::Task*,
                                                  const std::vector<Legion::PhysicalRegion>&,
                                                  Legion::Context,
                                                  Legion::Runtime*>;
      constexpr auto entry = T::BASE::template task_wrapper_<RET, variant_impl, variant_kind>;

      task_info->add_variant_(legate::TaskInfo::AddVariantKey{},
                              lib,
                              variant_kind,
                              variant_impl,
                              entry,
                              task_config,
                              options,
                              all_options);
    }
  }
};

}  // namespace legate::detail
