/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/geometry/domain_point.h>
#include <legate/utilities/assert.h>

#include <utility>

namespace legate {

inline DomainPoint::DomainPoint(std::int64_t index) { coordinates_.push_back(index); }

template <int DIM, typename T>
DomainPoint::DomainPoint(const Realm::Point<DIM, T>& point)
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  coordinates_.reserve(DIM);
  for (std::int32_t idx = 0; idx < DIM; ++idx) {
    coordinates_.push_back(static_cast<std::int64_t>(point[idx]));
  }
}

template <int DIM, typename T>
DomainPoint::operator Realm::Point<DIM, T>() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  LEGATE_ASSERT(DIM == static_cast<int>(dim()));

  Realm::Point<DIM, T> result;

  for (std::int32_t idx = 0; idx < DIM; ++idx) {
    result[idx] = static_cast<T>(coordinates_[idx]);
  }
  return result;
}

template <int DIM, typename T>
DomainPoint& DomainPoint::operator=(const Realm::Point<DIM, T>& point)
{
  *this = DomainPoint{point};
  return *this;
}

inline std::uint32_t DomainPoint::dim() const
{
  return static_cast<std::int32_t>(coordinates_.size());
}

template <typename Index, typename>
std::int64_t& DomainPoint::operator[](Index index)
{
  return coordinates_.at(static_cast<std::size_t>(index));
}

template <typename Index, typename>
const std::int64_t& DomainPoint::operator[](Index index) const
{
  return coordinates_.at(static_cast<std::size_t>(index));
}

inline const std::int64_t* DomainPoint::begin() const { return coordinates_.data(); }

inline const std::int64_t* DomainPoint::end() const
{
  return coordinates_.data() + coordinates_.size();
}

}  // namespace legate
