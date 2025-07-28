/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/task_signature.h>

#include <legate/partitioning/proxy.h>
#include <legate/task/detail/task_signature.h>
#include <legate/utilities/detail/small_vector.h>

#include <optional>

namespace legate {

TaskSignature::TaskSignature() : pimpl_{legate::make_shared<detail::TaskSignature>()} {}

TaskSignature::TaskSignature(InternalSharedPtr<detail::TaskSignature> impl)
  : pimpl_{std::move(impl)}
{
}

namespace {

[[nodiscard]] std::optional<detail::TaskSignature::Nargs> to_single_narg(std::uint32_t n)
{
  if (n == TaskSignature::UNBOUNDED) {
    return std::nullopt;
  }
  return detail::TaskSignature::Nargs{n};
}

}  // namespace

TaskSignature& TaskSignature::inputs(std::uint32_t n) noexcept
{
  impl_()->inputs(to_single_narg(n));
  return *this;
}

TaskSignature& TaskSignature::inputs(std::uint32_t low_bound, std::uint32_t upper_bound)
{
  impl_()->inputs({{low_bound, upper_bound}});
  return *this;
}

TaskSignature& TaskSignature::outputs(std::uint32_t n) noexcept
{
  impl_()->outputs(to_single_narg(n));
  return *this;
}

TaskSignature& TaskSignature::outputs(std::uint32_t low_bound, std::uint32_t upper_bound)
{
  impl_()->outputs({{low_bound, upper_bound}});
  return *this;
}

TaskSignature& TaskSignature::scalars(std::uint32_t n) noexcept
{
  impl_()->scalars(to_single_narg(n));
  return *this;
}

TaskSignature& TaskSignature::scalars(std::uint32_t low_bound, std::uint32_t upper_bound)
{
  impl_()->scalars({{low_bound, upper_bound}});
  return *this;
}

TaskSignature& TaskSignature::redops(std::uint32_t n) noexcept
{
  impl_()->redops(to_single_narg(n));
  return *this;
}

TaskSignature& TaskSignature::redops(std::uint32_t low_bound, std::uint32_t upper_bound)
{
  impl_()->redops({{low_bound, upper_bound}});
  return *this;
}

TaskSignature& TaskSignature::constraints(std::optional<Span<const ProxyConstraint>> constraints)
{
  std::optional<detail::SmallVector<InternalSharedPtr<detail::ProxyConstraint>>> ret = std::nullopt;

  if (constraints.has_value()) {
    auto& vec = ret.emplace();

    vec.reserve(constraints->size());
    for (auto&& c : *constraints) {
      vec.emplace_back(c.impl());
    }
  }
  impl_()->constraints(std::move(ret));
  return *this;
}

bool operator==(const TaskSignature& lhs, const TaskSignature& rhs) noexcept
{
  return *lhs.impl() == *rhs.impl();
}

bool operator!=(const TaskSignature& lhs, const TaskSignature& rhs) noexcept
{
  return *lhs.impl() != *rhs.impl();
}

TaskSignature::~TaskSignature() = default;

}  // namespace legate
