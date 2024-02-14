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

#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition fo legate::VariantOptions
 */
namespace legate {

// Each scalar output store can take up to 12 bytes, so in the worst case there can be only up to
// 341 scalar output stores.
constexpr std::size_t LEGATE_MAX_SIZE_SCALAR_RETURN = 4096;

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
  std::size_t return_size{LEGATE_MAX_SIZE_SCALAR_RETURN};

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
  VariantOptions& with_return_size(std::size_t return_size);

  void populate_registrar(Legion::TaskVariantRegistrar& registrar) const;
};

std::ostream& operator<<(std::ostream& os, const VariantOptions& options);

}  // namespace legate
