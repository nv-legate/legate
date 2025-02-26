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

#include <string_view>

namespace legate::detail {

class TaskSignature;
class AutoTask;

/**
 * @brief The base proxy constraint class which dictates the common interface. All proxy
 * constraints must derive from this.
 */
class ProxyConstraint {
 public:
  /**
   * @brief Validate the constraint arguments against the given signature.
   *
   * This call may throw any number of exceptions as part of the validation. This routine is
   * usually called when a signature is registered with a task.
   *
   * @param task_name The name of the task this constraint (and signature) corresponds to.
   * @param signature The signature of the task.
   */
  virtual void validate(std::string_view task_name, const TaskSignature& signature) const = 0;

  /**
   * @brief Apply the constraints on a task.
   *
   * The constraint should already have been validated by this point.
   *
   * @param task The task to apply the constraints to.
   */
  virtual void apply(AutoTask* task) const = 0;

  /**
   * @return The name of the constraint.
   */
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  [[nodiscard]] virtual bool operator==(const ProxyConstraint&) const = 0;
  [[nodiscard]] virtual bool operator!=(const ProxyConstraint& rhs) const;
};

[[nodiscard]] inline bool ProxyConstraint::operator!=(const ProxyConstraint& rhs) const
{
  return !(*this == rhs);
}

}  // namespace legate::detail
