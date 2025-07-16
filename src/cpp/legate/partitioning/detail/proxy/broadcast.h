/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/constraint.h>
#include <legate/partitioning/proxy.h>
#include <legate/utilities/detail/small_vector.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>

namespace legate::detail {

class AutoTask;
class TaskSignature;

/**
 * @brief The private broadcast constraint proxy.
 */
class ProxyBroadcast final : public ProxyConstraint {
 public:
  using value_type = std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>;

  /**
   * @brief Construct a broadcast constraint.
   *
   * @param value The value to broadcast.
   * @param axes The (possibly null) axes of value to broadcast.
   */
  ProxyBroadcast(value_type value,
                 std::optional<SmallVector<std::uint32_t, LEGATE_MAX_DIM>> axes) noexcept;

  /**
   * @return The value to broadcast.
   */
  [[nodiscard]] constexpr const value_type& value() const noexcept;

  /**
   * @return The (possibly null) axes to broadcast of `value()`.
   */
  [[nodiscard]] constexpr const std::optional<SmallVector<std::uint32_t, LEGATE_MAX_DIM>>& axes()
    const noexcept;

  /**
   * @return The name of the broadcast constraint.
   */
  [[nodiscard]] std::string_view name() const noexcept override;

  /**
   * @brief Validate the broadcast constraint.
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
  value_type value_;
  std::optional<SmallVector<std::uint32_t, LEGATE_MAX_DIM>> axes_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/proxy/broadcast.inl>
