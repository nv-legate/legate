/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/buffer.h>

#include <legate/data/detail/buffer.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>

#include <fmt/format.h>

#include <cstddef>
#include <stdexcept>

namespace legate::detail {

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

TaskLocalBuffer::TaskLocalBuffer(const TaskLocalBuffer&) = default;

TaskLocalBuffer& TaskLocalBuffer::operator=(const TaskLocalBuffer&) = default;

TaskLocalBuffer::TaskLocalBuffer(TaskLocalBuffer&&) noexcept = default;

TaskLocalBuffer& TaskLocalBuffer::operator=(TaskLocalBuffer&&) noexcept = default;

TaskLocalBuffer::~TaskLocalBuffer() = default;

TaskLocalBuffer::TaskLocalBuffer(SharedPtr<detail::TaskLocalBuffer> impl) : impl_{std::move(impl)}
{
}

TaskLocalBuffer::TaskLocalBuffer(const Legion::UntypedDeferredBuffer<>& buf,
                                 const Type& type,
                                 const Domain& bounds)
  : TaskLocalBuffer{legate::make_shared<detail::TaskLocalBuffer>(buf, type.impl(), bounds)}
{
}

TaskLocalBuffer::TaskLocalBuffer(const Type& type,
                                 Span<const std::uint64_t> bounds,
                                 std::optional<mapping::StoreTarget> mem_kind)
  : TaskLocalBuffer{{
                      type.size(),
                      static_cast<int>(bounds.size()),
                      mem_kind.has_value()
                        ? mapping::detail::to_kind(*mem_kind)
                        : find_memory_kind_for_executing_processor(/* host_accessible */ false),
                      detail::to_domain(bounds),
                    },
                    type,
                    detail::to_domain(bounds)}
{
}

Type TaskLocalBuffer::type() const { return Type{impl()->type()}; }

std::int32_t TaskLocalBuffer::dim() const { return impl()->dim(); }

const Domain& TaskLocalBuffer::domain() const { return impl()->domain(); }

mapping::StoreTarget TaskLocalBuffer::memory_kind() const { return impl()->memory_kind(); }

InlineAllocation TaskLocalBuffer::get_inline_allocation() const
{
  return impl()->get_inline_allocation();
}

const Legion::UntypedDeferredBuffer<>& TaskLocalBuffer::legion_buffer_() const
{
  return impl()->legion_buffer();
}

}  // namespace legate
