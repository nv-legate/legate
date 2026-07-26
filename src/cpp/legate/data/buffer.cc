/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/buffer.h>

#include <legate/data/detail/buffer.h>
#include <legate/data/detail/raw_allocation.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/machine.h>
#include <legate/utilities/shared_ptr.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace legate {

namespace detail {

void check_type(const legate::Type& ty, std::size_t size_of, std::size_t align_of)
{
  if (ty.size() != size_of) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Attempting to create buffer with incompatible type sizes. Expected {} (from "
                  "declared type {}), have {}",
                  ty.size(),
                  ty,
                  size_of)};
  }
  if (ty.alignment() != align_of) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Attempting to create buffer with incompatible type alignment. Expected {} (from "
                  "declared type {}), have {}",
                  ty.alignment(),
                  ty,
                  align_of)};
  }
}

void check_alignment(std::size_t alignment)
{
  if (alignment == 0) {
    throw TracedException<std::domain_error>{"alignment cannot be 0"};
  }

  constexpr auto is_power_of_2 = [](std::size_t n) { return (n & (n - 1)) == 0; };

  if (!is_power_of_2(alignment)) {
    throw TracedException<std::domain_error>{
      fmt::format("invalid alignment {}, must be a power of 2", alignment)};
  }
}

void throw_out_of_bounds(const Rect& domain, const Rect& to_access)
{
  throw TracedException<std::invalid_argument>{
    fmt::format("Domain for the pointer access ({}) is not contained in the buffer's domain ({})",
                to_access,
                domain)};
}

}  // namespace detail

Buffer::Buffer(const RawAllocation& allocation, const Type& type, Rect domain)
  : Buffer{legate::make_internal_shared<detail::Buffer>(
      allocation.impl(), type.impl(), std::move(domain))}
{
}

Type Buffer::type() const { return Type{impl()->type()}; }

const Rect& Buffer::domain() const { return impl()->domain(); }

std::uint32_t Buffer::dim() const { return impl()->dim(); }

mapping::StoreTarget Buffer::target() const { return impl()->target(); }

std::pair<void*, Point> Buffer::base_and_element_strides() const
{
  return impl()->base_and_element_strides();
}

RawAllocation Buffer::allocation() const
{
  return RawAllocation{InternalSharedPtr<detail::RawAllocation>{impl()->allocation()}};
}

void Buffer::reshape(Rect new_domain) { impl_->reshape(std::move(new_domain)); }

Buffer Buffer::allocate(const Type& type, Rect domain, mapping::StoreTarget target)
{
  return Buffer{detail::Buffer::allocate(type.impl(), std::move(domain), target)};
}

Buffer Buffer::allocate(const Type& type, Rect domain, Memory::Kind kind)
{
  return Buffer::allocate(type, std::move(domain), mapping::detail::to_target(kind));
}

Buffer::Buffer(const Buffer&) = default;

Buffer& Buffer::operator=(const Buffer&) = default;

Buffer::Buffer(Buffer&&) noexcept = default;

Buffer& Buffer::operator=(Buffer&&) noexcept = default;

Buffer::~Buffer() = default;

Buffer::Buffer(InternalSharedPtr<detail::Buffer> impl) : impl_{std::move(impl)} {}

}  // namespace legate
