/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/shape.h>

namespace legate {

inline Shape::Shape() : Shape{tuple<std::uint64_t>{}} {}

inline Shape::Shape(std::vector<std::uint64_t> extents)
  : Shape{tuple<std::uint64_t>{std::move(extents)}}
{
}

inline Shape::Shape(std::initializer_list<std::uint64_t> extents)
  : Shape{tuple<std::uint64_t>{std::move(extents)}}
{
}

inline std::uint64_t Shape::operator[](std::uint32_t idx) const { return extents()[idx]; }

inline std::uint64_t Shape::at(std::uint32_t idx) const { return extents().at(idx); }

inline bool Shape::operator!=(const Shape& other) const { return !operator==(other); }

inline Shape::Shape(InternalSharedPtr<detail::Shape> impl) : impl_{std::move(impl)} {}

inline const SharedPtr<detail::Shape>& Shape::impl() const { return impl_; }

}  // namespace legate
