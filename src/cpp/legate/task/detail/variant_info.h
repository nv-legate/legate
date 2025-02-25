/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/task/detail/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <legion/legion_types.h>

#include <string>

namespace legate::detail {

/**
 * @brief A class to describe the various properties of a variant.
 */
class VariantInfo {
 public:
  VariantInfo() = default;

  static_assert(!is_pure_move_constructible_v<Legion::CodeDescriptor>,
                "Use by value and std::move for Legion::CodeDescriptor");
  /**
   * @brief Construct a VariantInfo
   *
   * @param body_ The task function pointer.
   * @param code_desc_ The Legion code descriptor object that will be registered with Legion.
   * @param options_ The variant options set by the user.
   */
  VariantInfo(VariantImpl body_,
              const Legion::CodeDescriptor& code_desc_,
              VariantOptions options_,
              std::optional<InternalSharedPtr<TaskSignature>> signature_);

  /**
   * @return Get a textual representation of the variant info.
   */
  [[nodiscard]] std::string to_string() const;

  /**
   * @brief The task body
   */
  VariantImpl body{};

  /**
   * @brief The code descriptor to be registered.
   */
  Legion::CodeDescriptor code_desc{};

  /**
   * @brief The variant options.
   */
  VariantOptions options{};

  /**
   * @brief The task signature.
   */
  std::optional<InternalSharedPtr<TaskSignature>> signature{};
};

}  // namespace legate::detail

#include <legate/task/detail/variant_info.inl>
