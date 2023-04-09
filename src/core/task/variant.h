/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "legion.h"

#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition fo legate::VariantOptions
 */
namespace legate {

/**
 * @brief Function signature for task variants. Each task variant must be a function of this type.
 */
class TaskContext;
using VariantImpl = void (*)(TaskContext&);

// Each scalar output store can take up to 12 bytes, so in the worst case there can be only up to
// 341 scalar output stores.
constexpr size_t LEGATE_MAX_SIZE_SCALAR_RETURN = 4096;

/**
 * @ingroup task
 * @brief A helper class for specifying variant options
 */
struct VariantOptions {
  /**
   * @brief If the flag is `true`, the variant launches no subtasks. `true` by default.
   */
  bool leaf{true};
  bool inner{false};
  bool idempotent{false};
  /**
   * @brief If the flag is `true`, the variant needs a concurrent task launch. `false` by default.
   */
  bool concurrent{false};
  /**
   * @brief Maximum aggregate size for scalar output values. 4096 by default.
   */
  size_t return_size{LEGATE_MAX_SIZE_SCALAR_RETURN};

  /**
   * @brief Changes the value of the `leaf` flag
   *
   * @param `leaf` A new value for the `leaf` flag
   */
  VariantOptions& with_leaf(bool leaf);
  VariantOptions& with_inner(bool inner);
  VariantOptions& with_idempotent(bool idempotent);
  /**
   * @brief Changes the value of the `concurrent` flag
   *
   * @param `concurrent` A new value for the `concurrent` flag
   */
  VariantOptions& with_concurrent(bool concurrent);
  /**
   * @brief Sets a maximum aggregate size for scalar output values
   *
   * @param `return_size` A new maximum aggregate size for scalar output values
   */
  VariantOptions& with_return_size(size_t return_size);

  void populate_registrar(Legion::TaskVariantRegistrar& registrar);
};

std::ostream& operator<<(std::ostream& os, const VariantOptions& options);

namespace detail {

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

template <typename T, template <typename...> typename SELECTOR, bool HAS_VARIANT>
struct RegisterVariantImpl {
  static void register_variant(const VariantOptions& options)
  {
    T::BASE::template register_variant<SELECTOR<T>::variant>(SELECTOR<T>::id, options);
  }
};

template <typename T, template <typename...> typename SELECTOR>
struct RegisterVariantImpl<T, SELECTOR, false> {
  static void register_variant(const VariantOptions& options)
  {
    // Do nothing
  }
};

template <typename T, template <typename...> typename SELECTOR>
struct RegisterVariant {
  static void register_variant(const VariantOptions& options)
  {
    RegisterVariantImpl<T, SELECTOR, SELECTOR<T>::value>::register_variant(options);
  }
};

}  // namespace detail

}  // namespace legate
