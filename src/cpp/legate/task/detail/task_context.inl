/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/task_context.h>

#include <utility>

namespace legate::detail {

inline Span<const InternalSharedPtr<PhysicalArray>> TaskContext::inputs() const noexcept
{
  return inputs_;
}

inline Span<const InternalSharedPtr<PhysicalArray>> TaskContext::outputs() const noexcept
{
  return outputs_;
}

inline Span<const InternalSharedPtr<PhysicalArray>> TaskContext::reductions() const noexcept
{
  return reductions_;
}

inline Span<const InternalSharedPtr<Scalar>> TaskContext::scalars() const noexcept
{
  return scalars_;
}

inline Span<const legate::comm::Communicator> TaskContext::communicators() const noexcept
{
  return comms_;
}

inline VariantCode TaskContext::variant_kind() const noexcept { return variant_kind_; }

inline bool TaskContext::can_raise_exception() const noexcept { return can_raise_exception_; }

inline bool TaskContext::can_elide_device_ctx_sync() const noexcept
{
  return can_elide_device_ctx_sync_;
}

inline void TaskContext::set_exception(ReturnedException what) { excn_ = std::move(what); }

inline std::optional<ReturnedException>& TaskContext::get_exception() noexcept { return excn_; }

// ===========================================================================================

inline Span<const InternalSharedPtr<PhysicalStore>> TaskContext::get_unbound_stores_()
  const noexcept
{
  return unbound_stores_;
}

inline Span<const InternalSharedPtr<PhysicalStore>> TaskContext::get_scalar_stores_() const noexcept
{
  return scalar_stores_;
}

}  // namespace legate::detail
