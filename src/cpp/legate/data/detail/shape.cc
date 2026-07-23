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

Shape::Shape(const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& extents)
  : state_{State::BOUND}, dim_{static_cast<std::uint32_t>(extents.size())}, extents_(extents)
{
}

const Point& Shape::extents() const
{
  check_binding_();
  // `extents_` stores the extents as (non-negative) signed coordinates; expose them as the
  // unsigned extents the rest of the runtime expects. Reading signed storage through its unsigned
  // counterpart is well-defined, and the round-trip is lossless for valid extents.
  return extents_;
}

Rect Shape::domain() const
{
  const auto dim = static_cast<std::int32_t>(dim_);

  check_binding_();
  return {Point::zeroes(dim), extents_ - Point::full(dim, /* coord */ 1)};
}

void Shape::bind_extents(Point extents)
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

bool Shape::operator==(const Shape& other) const
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

void Shape::check_binding_() const
{
  if (State::UNBOUND == state_) {
    throw TracedException<std::invalid_argument>{
      "Illegal to access an uninitialized unbound store"};
  }
}

}  // namespace legate::detail
