/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/shape.h>

#include <legate/data/detail/shape.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace legate {

Shape::Shape(tuple<std::uint64_t> extents)
  : impl_{make_internal_shared<detail::Shape>(std::move(extents))}
{
}

const tuple<std::uint64_t>& Shape::extents() const { return impl()->extents(); }

std::size_t Shape::volume() const { return impl()->volume(); }

std::uint32_t Shape::ndim() const { return impl()->ndim(); }

std::string Shape::to_string() const { return impl()->to_string(); }

bool Shape::operator==(const Shape& other) const { return *impl() == *other.impl(); }

}  // namespace legate
