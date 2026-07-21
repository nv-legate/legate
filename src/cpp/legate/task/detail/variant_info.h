/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <string>

namespace legate::detail {

/**
 * @brief A class to describe the various properties of a variant.
 */
class VariantInfo {
 public:
  VariantInfo() = default;

  /**
   * @brief Construct a VariantInfo
   *
   * @param body_ The task function pointer.
   * @param options_ The variant options set by the user.
   * @param signature_ The task signature, or `std::nullopt` if the variant has none.
   */
  VariantInfo(VariantImpl body_,
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
