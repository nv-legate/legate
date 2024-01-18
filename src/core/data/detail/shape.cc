/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/data/detail/shape.h"

#include "core/data/shape.h"
#include "core/runtime/detail/runtime.h"
#include "core/utilities/detail/tuple.h"

namespace legate::detail {

Shape::Shape(tuple<uint64_t>&& extents)
  : state_{State::READY}, dim_{static_cast<uint32_t>(extents.size())}, extents_{std::move(extents)}
{
}

const tuple<uint64_t>& Shape::extents()
{
  switch (state_) {
    case State::UNBOUND: {
      ensure_binding();
      [[fallthrough]];
    }
    case State::BOUND: {
      const auto runtime = Runtime::get_runtime();
      auto domain        = runtime->get_index_space_domain(index_space_);
      extents_           = from_domain(domain);
      state_             = State::READY;
      runtime->find_or_create_region_manager(index_space_)->update_field_manager_match_credits();
      break;
    }
    case State::READY: {
      break;
    }
  }
  return extents_;
}

const Legion::IndexSpace& Shape::index_space()
{
  ensure_binding();
  if (!index_space_.exists()) {
    assert(State::READY == state_);
    index_space_ = Runtime::get_runtime()->find_or_create_index_space(extents_);
  }
  return index_space_;
}

void Shape::set_index_space(const Legion::IndexSpace& index_space)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(State::UNBOUND == state_);
  }
  index_space_ = index_space;
  state_       = State::BOUND;
}

std::string Shape::to_string() const
{
  switch (state_) {
    case State::UNBOUND: {
      return "Shape(unbound " + std::to_string(dim_) + "D)";
    }
    case State::BOUND: {
      return "Shape(bound " + std::to_string(dim_) + "D)";
    }
    case State::READY: {
      return "Shape" + extents_.to_string();
    }
  }
  assert(false);
  return "";
}

bool Shape::operator==(Shape& other)
{
  if (this == &other) {
    return true;
  }
  if (State::UNBOUND == state_ || State::UNBOUND == other.state_) {
    Runtime::get_runtime()->flush_scheduling_window();
    if (State::UNBOUND == state_ || State::UNBOUND == other.state_) {
      throw std::invalid_argument{"Illegal to access an uninitialized unbound store"};
    }
  }
  // If both shapes are in the bound state and their index spaces are the same, we can elide the
  // blocking equivalence check
  if (State::BOUND == state_ && State::BOUND == other.state_ &&
      index_space_ == other.index_space_) {
    return true;
  }
  // Otherwise, we have no choice but block waiting on the exact extents
  return extents() == other.extents();
}

void Shape::ensure_binding()
{
  if (State::UNBOUND != state_) {
    return;
  }
  Runtime::get_runtime()->flush_scheduling_window();
  if (State::UNBOUND == state_) {
    throw std::invalid_argument{"Illegal to access an uninitialized unbound store"};
  }
}

}  // namespace legate::detail
