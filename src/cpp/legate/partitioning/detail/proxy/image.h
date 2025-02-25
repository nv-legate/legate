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

#include <legate/partitioning/detail/proxy/constraint.h>
#include <legate/partitioning/proxy.h>

#include <optional>
#include <string_view>
#include <variant>

namespace legate::detail {

class AutoTask;
class TaskSignature;

}  // namespace legate::detail

namespace legate::detail::proxy {

/**
 * @brief The private image constraint proxy.
 */
class Image final : public Constraint {
 public:
  using value_type = std::variant<legate::proxy::ArrayArgument,
                                  legate::proxy::InputArguments,
                                  legate::proxy::OutputArguments,
                                  legate::proxy::ReductionArguments>;

  /**
   * @brief Construct an image constraint.
   *
   * @param var_function The function variable.
   * @param var_range The range variable.
   * @param hint The (possibly null) hint.
   */
  Image(value_type var_function,
        value_type var_range,
        std::optional<ImageComputationHint> hint) noexcept;

  /**
   * @return The function variable.
   */
  [[nodiscard]] constexpr const value_type& var_function() const noexcept;

  /**
   * @return The range variable.
   */
  [[nodiscard]] constexpr const value_type& var_range() const noexcept;

  /**
   * @return The (possibly null) image hint.
   */
  [[nodiscard]] constexpr const std::optional<ImageComputationHint>& hint() const noexcept;

  /**
   * @return The name of the image constraint.
   */
  [[nodiscard]] std::string_view name() const noexcept override;

  /**
   * @brief Validate the image constraint.
   *
   * @param task_name The name of the task this constraint (and signature) corresponds to.
   * @param signature The signature of the task.
   */
  void validate(std::string_view task_name, const TaskSignature& signature) const override;

  /**
   * @brief Apply the constraints on a task.
   *
   * The constraint should already have been validated by this point.
   *
   * @param task The task to apply the constraints to.
   */
  void apply(AutoTask* task) const override;

  [[nodiscard]] bool operator==(const Constraint& rhs) const noexcept override;

 private:
  value_type var_function_;
  value_type var_range_;
  std::optional<ImageComputationHint> hint_{};
};

}  // namespace legate::detail::proxy

#include <legate/partitioning/detail/proxy/image.inl>
