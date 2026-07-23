/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/buffer.h>

#include <legate/data/detail/buffer.h>
#include <legate/data/detail/inline_storage.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/machine.h>
#include <legate/utilities/shared_ptr.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace legate::detail {

Rect to_rect(Span<const std::uint64_t> shape)
{
  const auto ndim = static_cast<std::int32_t>(shape.size());
  auto lo         = Point::zeroes(ndim);
  auto hi         = Point::zeroes(ndim);

  for (std::int32_t idx = 0; idx < ndim; ++idx) {
    hi[idx] = static_cast<std::int64_t>(shape[idx]) - 1;
  }
  return {std::move(lo), std::move(hi)};
}

void check_alignment(std::size_t alignment)
{
  if (alignment == 0) {
    throw detail::TracedException<std::domain_error>{"alignment cannot be 0"};
  }

  constexpr auto is_power_of_2 = [](std::size_t n) { return (n & (n - 1)) == 0; };

  if (!is_power_of_2(alignment)) {
    throw detail::TracedException<std::domain_error>{
      fmt::format("invalid alignment {}, must be a power of 2", alignment)};
  }
}

InternalSharedPtr<InlineStorage> create_buffer_storage(std::uint32_t field_size,
                                                       const Rect& bounds,
                                                       Memory::Kind kind)
{
  return make_internal_shared<InlineStorage>(bounds, field_size, mapping::detail::to_target(kind));
}

void inline_storage_layout(const InternalSharedPtr<InlineStorage>& storage,
                           void** base_out,
                           std::uint64_t* element_strides_out)
{
  auto&& [base, element_strides] = storage->base_and_element_strides();

  *base_out = base;
  for (std::uint32_t i = 0; i < element_strides.dim(); ++i) {
    element_strides_out[i] = static_cast<std::uint64_t>(element_strides[i]);
  }
}

Memory::Kind get_memory_kind(const InternalSharedPtr<InlineStorage>& storage)
{
  return mapping::detail::to_kind(storage->target());
}

}  // namespace legate::detail

namespace legate {

namespace untyped_buffer_detail {

void check_type(const Type& ty, std::size_t size_of, std::size_t align_of)
{
  if (ty.size() != size_of) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Attempting to create buffer with incompatible type sizes. Expected {} (from "
                  "declared type {}), have {}",
                  ty.size(),
                  ty,
                  size_of)};
  }
  if (ty.alignment() != align_of) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Attempting to create buffer with incompatible type alignment. Expected {} (from "
                  "declared type {}), have {}",
                  ty.alignment(),
                  ty,
                  align_of)};
  }
}

}  // namespace untyped_buffer_detail

namespace {

// Allocates an inline storage for a task-local buffer, resolving the memory kind from the
// executing processor when one is not explicitly requested.
[[nodiscard]] InternalSharedPtr<detail::InlineStorage> make_task_local_inline_storage(
  const Type& type, Span<const std::uint64_t> bounds, std::optional<mapping::StoreTarget> mem_kind)
{
  const auto target = mem_kind.value_or(mapping::detail::to_target(
    find_memory_kind_for_executing_processor(/* host_accessible */ false)));

  return make_internal_shared<detail::InlineStorage>(detail::to_rect(bounds), type.size(), target);
}

}  // namespace

TaskLocalBuffer::TaskLocalBuffer(const TaskLocalBuffer&) = default;

TaskLocalBuffer& TaskLocalBuffer::operator=(const TaskLocalBuffer&) = default;

TaskLocalBuffer::TaskLocalBuffer(TaskLocalBuffer&&) noexcept = default;

TaskLocalBuffer& TaskLocalBuffer::operator=(TaskLocalBuffer&&) noexcept = default;

TaskLocalBuffer::~TaskLocalBuffer() = default;

TaskLocalBuffer::TaskLocalBuffer(SharedPtr<detail::TaskLocalBuffer> impl) : impl_{std::move(impl)}
{
}

TaskLocalBuffer::TaskLocalBuffer(InternalSharedPtr<detail::InlineStorage> storage,
                                 const Type& type,
                                 // Rect is a POD type, no point in moving it
                                 const Rect& bounds)  // NOLINT(modernize-pass-by-value)
  : TaskLocalBuffer{
      legate::make_shared<detail::TaskLocalBuffer>(std::move(storage), type.impl(), bounds)}
{
}

TaskLocalBuffer::TaskLocalBuffer(const Type& type,
                                 Span<const std::uint64_t> bounds,
                                 std::optional<mapping::StoreTarget> mem_kind)
  : TaskLocalBuffer{
      make_task_local_inline_storage(type, bounds, mem_kind), type, detail::to_rect(bounds)}
{
}

Type TaskLocalBuffer::type() const { return Type{impl()->type()}; }

std::int32_t TaskLocalBuffer::dim() const { return static_cast<std::int32_t>(impl()->dim()); }

const Rect& TaskLocalBuffer::domain() const { return impl()->domain(); }

mapping::StoreTarget TaskLocalBuffer::memory_kind() const { return impl()->memory_kind(); }

InlineAllocation TaskLocalBuffer::get_inline_allocation() const
{
  return impl()->get_inline_allocation();
}

const InternalSharedPtr<detail::InlineStorage>& TaskLocalBuffer::storage_() const
{
  return impl()->get_storage();
}

}  // namespace legate
