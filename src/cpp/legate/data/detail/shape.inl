/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/shape.h>

namespace legate::detail {

inline Shape::Shape(std::uint32_t dim) : dim_{dim} {}

inline bool Shape::unbound() const { return state_ == State::UNBOUND; }

inline bool Shape::ready() const { return state_ == State::READY; }

inline std::uint32_t Shape::ndim() const { return dim_; }

inline std::size_t Shape::volume() { return extents().volume(); }

inline bool Shape::operator!=(Shape& other) { return !operator==(other); }

}  // namespace legate::detail
