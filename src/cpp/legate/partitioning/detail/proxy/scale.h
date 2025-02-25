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
#include <legate/utilities/tuple.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace legate::detail {

class AutoTask;
class TaskSignature;

}  // namespace legate::detail

namespace legate::detail::proxy {

/**
 * @brief The private scaling constraint.
 */
class Scale final : public Constraint {
 public:
  using value_type = std::variant<legate::proxy::ArrayArgument,
                                  legate::proxy::InputArguments,
                                  legate::proxy::OutputArguments,
                                  legate::proxy::ReductionArguments>;

  /**
   * @brief Construct a scaling constraint.
   *
   * @param factors The factors to scale the various axes with.
   * @param var_smaller The variable to scale.
   * @param var_bigger The variable to scale to.
   */
  Scale(tuple<std::uint64_t> factors, value_type var_smaller, value_type var_bigger) noexcept;

  /**
   * @return The scaling factors.
   */
  [[nodiscard]] constexpr const tuple<std::uint64_t>& factors() const noexcept;

  /**
   * @return The variable to scale.
   */
  [[nodiscard]] constexpr const value_type& var_smaller() const noexcept;

  /**
   * @return The variable to scale to.
   */
  [[nodiscard]] constexpr const value_type& var_bigger() const noexcept;

  /**
   * @return The name of this scaling constraint.
   */
  [[nodiscard]] std::string_view name() const noexcept override;

  /**
   * @brief Validate the scaling constraint.
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
  tuple<std::uint64_t> factors_{};
  value_type var_smaller_;
  value_type var_bigger_;
};

}  // namespace legate::detail::proxy

#include <legate/partitioning/detail/proxy/scale.inl>
