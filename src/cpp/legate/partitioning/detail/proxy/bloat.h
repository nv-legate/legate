/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/constraint.h>
#include <legate/partitioning/proxy.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/span.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace legate::detail {

class AutoTask;
class TaskSignature;

/**
 * @brief The private bloat constraint proxy.
 */
class ProxyBloat final : public ProxyConstraint {
 public:
  using value_type = std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>;

  /**
   * @brief Construct a bloat constraint.
   *
   * @param var_source The source variable.
   * @param var_bloat The variable to bloat.
   * @param low_offsets The lower offsets.
   * @param high_offsets The high offsets.
   */
  ProxyBloat(value_type var_source,
             value_type var_bloat,
             SmallVector<std::uint64_t, LEGATE_MAX_DIM> low_offsets,
             SmallVector<std::uint64_t, LEGATE_MAX_DIM> high_offsets) noexcept;

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
  [[nodiscard]] constexpr Span<const std::uint64_t> low_offsets() const noexcept;

  /**
   * @return The high offsets.
   */
  [[nodiscard]] constexpr Span<const std::uint64_t> high_offsets() const noexcept;

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

  [[nodiscard]] bool operator==(const ProxyConstraint& rhs) const override;

 private:
  value_type var_source_;
  value_type var_bloat_;
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> low_offsets_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> high_offsets_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/proxy/bloat.inl>
