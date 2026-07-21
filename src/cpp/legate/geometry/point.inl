/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/geometry/point.h>
#include <legate/utilities/assert.h>

#include <utility>

namespace legate {

inline Point::Point(std::int64_t index) { coordinates_.push_back(index); }

inline Point::Point(detail::SmallVector<std::int64_t, LEGATE_MAX_DIM> coords)
  : coordinates_{std::move(coords)}
{
}

inline Point::Point(Span<const std::int64_t> coords) : coordinates_{coords} {}

template <int DIM>
Point::Point(const NDPoint<DIM>& point)
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  coordinates_.reserve(DIM);
  for (std::int32_t idx = 0; idx < DIM; ++idx) {
    coordinates_.push_back(static_cast<std::int64_t>(point[idx]));
  }
}

template <int DIM>
Point::operator NDPoint<DIM>() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  LEGATE_ASSERT(DIM == static_cast<int>(dim()));

  NDPoint<DIM> result;

  for (std::int32_t idx = 0; idx < DIM; ++idx) {
    result[idx] = coordinates_[idx];
  }
  return result;
}

template <int DIM>
Point& Point::operator=(const NDPoint<DIM>& point)
{
  *this = Point{point};
  return *this;
}

inline std::uint32_t Point::dim() const { return static_cast<std::int32_t>(coordinates_.size()); }

inline std::size_t Point::size() const { return coordinates_.size(); }

inline bool Point::empty() const { return coordinates_.empty(); }

template <typename Index, typename>
std::int64_t& Point::operator[](Index index)
{
  return coordinates_.at(static_cast<std::size_t>(index));
}

template <typename Index, typename>
const std::int64_t& Point::operator[](Index index) const
{
  return coordinates_.at(static_cast<std::size_t>(index));
}

/*static*/ inline Point Point::zeroes(std::int32_t ndim) { return full(ndim, 0); }

inline const std::int64_t* Point::begin() const { return coordinates_.data(); }

inline const std::int64_t* Point::end() const { return coordinates_.data() + coordinates_.size(); }

}  // namespace legate
