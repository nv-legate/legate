/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

/**
 * @file
 * @brief Class definition of legate::VariantInfo.
 */

namespace legate::detail {

class VariantInfo;

}  // namespace legate::detail

namespace legate {

class VariantOptions;

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief A class describing the various properties of a task variant.
 */
class VariantInfo {
 public:
  VariantInfo() = delete;

  explicit VariantInfo(const detail::VariantInfo& impl) noexcept;

  /**
   * @return Get the variant options sets for this variant.
   *
   * @see VariantOptions
   */
  [[nodiscard]] const VariantOptions& options() const noexcept;

 private:
  [[nodiscard]] const detail::VariantInfo& impl_() const noexcept;

  const detail::VariantInfo* pimpl_{};
};

/** @} */

}  // namespace legate

#include <legate/task/variant_info.inl>
