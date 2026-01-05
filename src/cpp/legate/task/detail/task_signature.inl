/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/task_signature.h>

namespace legate::detail {

inline const std::variant<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>&
TaskSignature::Nargs::value() const
{
  return value_;
}

// ------------------------------------------------------------------------------------------

constexpr void TaskSignature::inputs(std::optional<Nargs> n) noexcept
{
  num_inputs_ = std::move(n);
}

constexpr void TaskSignature::outputs(std::optional<Nargs> n) noexcept
{
  num_outputs_ = std::move(n);
}

constexpr void TaskSignature::scalars(std::optional<Nargs> n) noexcept
{
  num_scalars_ = std::move(n);
}

constexpr void TaskSignature::redops(std::optional<Nargs> n) noexcept
{
  num_redops_ = std::move(n);
}

constexpr const std::optional<TaskSignature::Nargs>& TaskSignature::inputs() const noexcept
{
  return num_inputs_;
}

constexpr const std::optional<TaskSignature::Nargs>& TaskSignature::outputs() const noexcept
{
  return num_outputs_;
}

constexpr const std::optional<TaskSignature::Nargs>& TaskSignature::scalars() const noexcept
{
  return num_scalars_;
}

constexpr const std::optional<TaskSignature::Nargs>& TaskSignature::redops() const noexcept
{
  return num_redops_;
}

inline std::optional<Span<const InternalSharedPtr<detail::ProxyConstraint>>>
TaskSignature::constraints() const noexcept
{
  return constraints_;
}

}  // namespace legate::detail
