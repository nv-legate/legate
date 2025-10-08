/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
  : state_{State::READY},
    dim_{static_cast<std::uint32_t>(extents.size())},
    extents_{std::move(extents)}
{
}

Span<const std::uint64_t> Shape::extents()
{
  switch (state_) {
    case State::UNBOUND: {
      ensure_binding_();
      [[fallthrough]];
    }
    case State::BOUND: {
      auto&& runtime = Runtime::get_runtime();
      auto domain    = runtime.get_index_space_domain(index_space_);
      extents_       = from_domain(domain);
      state_         = State::READY;
      break;
    }
    case State::READY: {
      break;
    }
  }
  return extents_;
}

std::size_t Shape::volume() { return array_volume(extents()); }

const Legion::IndexSpace& Shape::index_space()
{
  ensure_binding_();
  if (!index_space_.exists()) {
    LEGATE_CHECK(State::READY == state_);
    index_space_ = Runtime::get_runtime().find_or_create_index_space(extents_);
  }
  return index_space_;
}

void Shape::set_index_space(const Legion::IndexSpace& index_space)
{
  LEGATE_CHECK(State::UNBOUND == state_);
  index_space_ = index_space;
  state_       = State::BOUND;
}

void Shape::copy_extents_from(const Shape& other)
{
  LEGATE_CHECK(State::BOUND == state_);
  LEGATE_ASSERT(dim_ == other.dim_);
  LEGATE_ASSERT(index_space_ == other.index_space_);
  state_   = State::READY;
  extents_ = other.extents_;
}

std::string Shape::to_string() const
{
  switch (state_) {
    case State::UNBOUND: {
      return fmt::format("Shape(unbound {}D)", dim_);
    }
    case State::BOUND: {
      return fmt::format("Shape(bound {}D)", dim_);
    }
    case State::READY: {
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
    Runtime::get_runtime().flush_scheduling_window();
    if (State::UNBOUND == state_ || State::UNBOUND == other.state_) {
      throw TracedException<std::invalid_argument>{
        "Illegal to access an uninitialized unbound store"};
    }
  }
  // If both shapes are in the bound state and their index spaces are the same, we can elide the
  // blocking equivalence check
  if (State::BOUND == state_ && State::BOUND == other.state_ &&
      index_space_ == other.index_space_) {
    return true;
  }
  // Otherwise, we have no choice but block waiting on the exact extents
  return std::equal(
    extents().begin(), extents().end(), other.extents().begin(), other.extents().end());
}

void Shape::ensure_binding_()
{
  if (State::UNBOUND != state_) {
    return;
  }
  Runtime::get_runtime().flush_scheduling_window();
  if (State::UNBOUND == state_) {
    throw TracedException<std::invalid_argument>{
      "Illegal to access an uninitialized unbound store"};
  }
}

}  // namespace legate::detail
