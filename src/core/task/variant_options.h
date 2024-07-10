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

#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition of legate::VariantOptions
 */
namespace legate {

// Each scalar output store can take up to 12 bytes, so in the worst case there can be only up to
// 341 scalar output stores.
inline constexpr std::size_t LEGATE_MAX_SIZE_SCALAR_RETURN = 4096;

/**
 * @brief A helper class for specifying variant options
 *
 * @ingroup task
 */
class VariantOptions {
 public:
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
  std::size_t return_size{LEGATE_MAX_SIZE_SCALAR_RETURN};

  /**
   * @brief Whether this variant can skip device context synchronization after completion.
   *
   * Normally, for device-enabled task variants, Legate will emit a device-wide barrier to
   * ensure that all outstanding (potentially asynchronous) work performed by the variant has
   * completed. However, if the task launches no such work, or if that work is launched using
   * the task-specific device streams, then such a context synchronization is not necessary.
   *
   * Setting this value to `true` ensures that no context synchronization is performed. Setting
   * it to `false` guarantees that a context synchronization is done.
   *
   * Has no effect on non-device variants (for example CPU variants).
   *
   * @see with_elide_device_ctx_sync()
   */
  bool elide_device_ctx_sync{};

  /**
   * @brief Changes the value of the `leaf` flag
   *
   * @param `leaf` A new value for the `leaf` flag
   */
  constexpr VariantOptions& with_leaf(bool leaf);
  constexpr VariantOptions& with_inner(bool inner);
  constexpr VariantOptions& with_idempotent(bool idempotent);
  /**
   * @brief Changes the value of the `concurrent` flag
   *
   * @param `concurrent` A new value for the `concurrent` flag
   */
  constexpr VariantOptions& with_concurrent(bool concurrent);
  /**
   * @brief Sets a maximum aggregate size for scalar output values
   *
   * @param `return_size` A new maximum aggregate size for scalar output values
   */
  constexpr VariantOptions& with_return_size(std::size_t return_size);

  /**
   * @brief Sets whether the variant can elide device context synchronization after task
   * completion.
   *
   * @param `elide_sync` `true` if this variant can skip synchronizing the device context after
   * task completion, `false` otherwise.
   *
   * @return reference to `this`.
   *
   * @see elide_device_ctx_sync
   */
  constexpr VariantOptions& with_elide_device_ctx_sync(bool elide_sync);

  /**
   * @brief Populate a Legion::TaskVariantRegistrar using the options contained.
   *
   * @param registrar The registrar to fill out.
   */
  void populate_registrar(Legion::TaskVariantRegistrar& registrar) const;

  [[nodiscard]] constexpr bool operator==(const VariantOptions& other) const;
  [[nodiscard]] constexpr bool operator!=(const VariantOptions& other) const;

  /**
   * @brief The default variant options used during task creation if no user-supplied options
   * are given.
   */
  static const VariantOptions DEFAULT_OPTIONS;
};

// This trick is needed because you cannot declare a constexpr variable of the same class
// inside the class definition, because at that point the class is still considered an
// incomplete type.
//
// Do not be fooled, DEFAULT_VARIANT_OPTIONS is still constexpr; for variables, constexpr can
// explicitly only be on a definition, not on any declarations
// (eel.is/c++draft/dcl.constexpr#1.sentence-1). The static const is the declaration, the line
// below is the definition.
inline constexpr VariantOptions VariantOptions::DEFAULT_OPTIONS{};

std::ostream& operator<<(std::ostream& os, const VariantOptions& options);

}  // namespace legate

#include "core/task/variant_options.inl"
