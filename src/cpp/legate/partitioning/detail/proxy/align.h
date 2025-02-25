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

#include <string_view>
#include <variant>

namespace legate::detail {

class AutoTask;
class TaskSignature;

}  // namespace legate::detail

namespace legate::detail::proxy {

/**
 * @brief The private alignment constraint proxy.
 */
class Align final : public Constraint {
 public:
  using value_type = std::variant<legate::proxy::ArrayArgument,
                                  legate::proxy::InputArguments,
                                  legate::proxy::OutputArguments,
                                  legate::proxy::ReductionArguments>;

  /**
   * @brief Construct and alignment constraint.
   *
   * @param left The lhs argument to align.
   * @param right The rhs argument to align.
   */
  Align(value_type left, value_type right) noexcept;

  /**
   * @return The left argument for the alignment constraint.
   */
  [[nodiscard]] constexpr const value_type& left() const noexcept;

  /**
   * @return The right argument for the alignment constraint.
   */
  [[nodiscard]] constexpr const value_type& right() const noexcept;

  /**
   * @return The name of the alignment constraint.
   */
  [[nodiscard]] std::string_view name() const noexcept override;

  /**
   * @brief Validate the alignment constraint.
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
  value_type left_;
  value_type right_;
};

}  // namespace legate::detail::proxy

#include <legate/partitioning/detail/proxy/align.inl>
