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
