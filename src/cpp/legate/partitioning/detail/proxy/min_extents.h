/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

namespace legate {

enum class ImageComputationHint : std::uint8_t;

}  // namespace legate

namespace legate::detail {

class AutoTask;
class TaskSignature;

/**
 * @brief The private image constraint proxy.
 */
class ProxyMinExtents final : public ProxyConstraint {
 public:
  using value_type = std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>;

  /**
   * @brief Construct an image constraint.
   *
   * @param variable The target variable.
   * @param minimum_extents The minimum extents.
   */
  ProxyMinExtents(value_type variable,
                  SmallVector<std::uint64_t, LEGATE_MAX_DIM> minimum_extents) noexcept;

  /**
   * @return The target variable.
   */
  [[nodiscard]] constexpr const value_type& variable() const noexcept;

  /**
   * @return The minimum extents.
   */
  [[nodiscard]] constexpr const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& minimum_extents()
    const noexcept;

  /**
   * @return The name of the minimum-extent constraint.
   */
  [[nodiscard]] std::string_view name() const noexcept override;

  /**
   * @brief Validate the minimum-extent constraint.
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
  value_type variable_;
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> minimum_extents_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/proxy/min_extents.inl>
