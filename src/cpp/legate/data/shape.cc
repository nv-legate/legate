/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/shape.h>

#include <legate/data/detail/shape.h>
#include <legate/utilities/detail/small_vector.h>

#include <cstddef>
#include <cstdint>

namespace legate {

Shape::Shape(Span<const std::uint64_t> extents)
  : impl_{make_internal_shared<detail::Shape>(
      detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{extents})}
{
}

Shape::Shape() : Shape{Span<const std::uint64_t>{}} {}

Shape::Shape(const tuple<std::uint64_t>& extents) : Shape{extents.data()} {}

Shape::Shape(const std::vector<std::uint64_t>& extents) : Shape{Span<const std::uint64_t>{extents}}
{
}

Shape::Shape(std::initializer_list<std::uint64_t> extents)
  : Shape{Span<const std::uint64_t>{extents}}
{
}

std::uint64_t Shape::operator[](std::uint32_t idx) const { return impl()->extents()[idx]; }

tuple<std::uint64_t> Shape::extents() const
{
  auto&& ext = impl()->extents();

  return tuple<std::uint64_t>{std::vector<std::uint64_t>{ext.begin(), ext.end()}};
}

std::uint64_t Shape::at(std::uint32_t idx) const { return impl()->extents().at(idx); }

std::size_t Shape::volume() const { return impl()->volume(); }

std::uint32_t Shape::ndim() const { return impl()->ndim(); }

std::string Shape::to_string() const { return impl()->to_string(); }

bool Shape::operator==(const Shape& other) const { return *impl() == *other.impl(); }

bool Shape::operator!=(const Shape& other) const { return *impl() != *other.impl(); }

}  // namespace legate
