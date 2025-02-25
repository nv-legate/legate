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
 * @brief The private bloat constraint proxy.
 */
class Bloat final : public Constraint {
 public:
  using value_type = std::variant<legate::proxy::ArrayArgument,
                                  legate::proxy::InputArguments,
                                  legate::proxy::OutputArguments,
                                  legate::proxy::ReductionArguments>;

  /**
   * @brief Construct a bloat constraint.
   *
   * @param var_source The source variable.
   * @param var_bloat The variable to bloat.
   * @param low_offsets The lower offsets.
   * @param high_offsets The high offsets.
   */
  Bloat(value_type var_source,
        value_type var_bloat,
        tuple<std::uint64_t> low_offsets,
        tuple<std::uint64_t> high_offsets) noexcept;

  /**
   * @return The source variable.
   */
  [[nodiscard]] constexpr const value_type& var_source() const noexcept;

  /**
   * @return The bloat variable.
   */
  [[nodiscard]] constexpr const value_type& var_bloat() const noexcept;

  /**
   * @return The low offsets.
   */
  [[nodiscard]] constexpr const tuple<std::uint64_t>& low_offsets() const noexcept;

  /**
   * @return The high offsets.
   */
  [[nodiscard]] constexpr const tuple<std::uint64_t>& high_offsets() const noexcept;

  /**
   * @return The name of the bloat constraint.
   */
  [[nodiscard]] std::string_view name() const noexcept override;

  /**
   * @brief Validate the bloat constraint.
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
  value_type var_source_;
  value_type var_bloat_;
  tuple<std::uint64_t> low_offsets_{};
  tuple<std::uint64_t> high_offsets_{};
};

}  // namespace legate::detail::proxy

#include <legate/partitioning/detail/proxy/bloat.inl>
