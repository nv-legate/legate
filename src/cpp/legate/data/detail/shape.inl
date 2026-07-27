/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/shape.h>

namespace legate::detail {

inline Shape::Shape(std::uint32_t dim) : dim_{dim} {}

inline Shape::Shape(Point extents)
  : state_{State::BOUND}, dim_{extents.dim()}, extents_{std::move(extents)}
{
}

inline bool Shape::unbound() const { return state_ == State::UNBOUND; }

inline std::uint32_t Shape::ndim() const { return dim_; }

inline bool Shape::operator!=(const Shape& other) const { return !operator==(other); }

}  // namespace legate::detail
