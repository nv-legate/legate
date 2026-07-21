/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/shape.h>

#include <legate/data/shape.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>

#include <fmt/ranges.h>

#include <stdexcept>

namespace legate::detail {

Shape::Shape(SmallVector<std::uint64_t, LEGATE_MAX_DIM>&& extents)
  : state_{State::BOUND},
    dim_{static_cast<std::uint32_t>(extents.size())},
    extents_{std::move(extents)}
{
}

Span<const std::uint64_t> Shape::extents()
{
  ensure_binding_();
  return extents_;
}

std::size_t Shape::volume() { return array_volume(extents()); }

void Shape::copy_extents_from(const Shape& other)
{
  LEGATE_CHECK(State::BOUND == state_);
  LEGATE_ASSERT(dim_ == other.dim_);
  extents_ = other.extents_;
}

void Shape::bind_extents(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents)
{
  LEGATE_ASSERT(dim_ == extents.size());
  state_   = State::BOUND;
  extents_ = std::move(extents);
}

std::string Shape::to_string() const
{
  switch (state_) {
    case State::UNBOUND: {
      return fmt::format("Shape(unbound {}D)", dim_);
    }
    case State::BOUND: {
      return fmt::format("Shape {}", extents_);
    }
  }
  return "";
}

bool Shape::operator==(Shape& other)
{
  if (this == &other) {
    return true;
  }
  if (State::UNBOUND == state_ || State::UNBOUND == other.state_) {
    throw TracedException<std::invalid_argument>{
      "Illegal to access an uninitialized unbound store"};
  }
  return std::equal(
    extents().begin(), extents().end(), other.extents().begin(), other.extents().end());
}

void Shape::ensure_binding_()
{
  if (State::UNBOUND == state_) {
    throw TracedException<std::invalid_argument>{
      "Illegal to access an uninitialized unbound store"};
  }
}

}  // namespace legate::detail
