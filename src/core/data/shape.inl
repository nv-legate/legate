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

#pragma once

#include "core/data/shape.inl"

namespace legate {

inline Shape::Shape() : Shape(tuple<std::uint64_t>{}) {}

inline Shape::Shape(std::vector<std::uint64_t> extents)
  : Shape{tuple<std::uint64_t>{std::move(extents)}}
{
}

inline Shape::Shape(std::initializer_list<std::uint64_t> extents)
  : Shape{tuple<std::uint64_t>{std::move(extents)}}
{
}

inline std::size_t Shape::volume() const { return extents().volume(); }

inline std::uint64_t Shape::operator[](std::uint32_t idx) const { return extents()[idx]; }

inline std::uint64_t Shape::at(std::uint32_t idx) const { return extents().at(idx); }

inline bool Shape::operator!=(const Shape& other) const { return !operator==(other); }

inline Shape::Shape(InternalSharedPtr<detail::Shape> impl) : impl_{std::move(impl)} {}

inline const SharedPtr<detail::Shape>& Shape::impl() const { return impl_; }

}  // namespace legate
