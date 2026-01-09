/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/constraint.h>
#include <legate/partitioning/proxy.h>

#include <string_view>
#include <variant>

namespace legate::detail {

class AutoTask;
class TaskSignature;

/**
 * @brief The private alignment constraint proxy.
 */
class ProxyAlign final : public ProxyConstraint {
 public:
  using value_type = std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>;

  /**
   * @brief Construct and alignment constraint.
   *
   * @param left The lhs argument to align.
   * @param right The rhs argument to align.
   */
  ProxyAlign(value_type left, value_type right) noexcept;

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

  [[nodiscard]] bool operator==(const ProxyConstraint& rhs) const override;

 private:
  value_type left_;
  value_type right_;
};

}  // namespace legate::detail

#include <legate/partitioning/detail/proxy/align.inl>
