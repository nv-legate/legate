/*
 *  * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 *   * reserved.
 *    * SPDX-License-Identifier: Apache-2.0
 *     */

#pragma once

#include <legate/partitioning/detail/restriction.h>

namespace legate::detail {

inline Restrictions::Restrictions(std::uint32_t ndim)
  : dim_restrictions_{tags::size_tag, ndim, Restriction::ALLOW},
    min_extents_{tags::size_tag, ndim, 0}
{
}

inline Restrictions::Restrictions(const SmallVector<Restriction>& dimension_restrictions)
  : Restrictions{
      dimension_restrictions,
      SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tags::size_tag, dimension_restrictions.size(), 0}}
{
}

inline void Restrictions::set_require_invertible(bool new_value) { req_invertible_ = new_value; }

template <typename RES_FUNC, typename EXT_FUNC>
Restrictions Restrictions::map(RES_FUNC&& f, EXT_FUNC&& g) &&
{
  return {f(std::move(dimension_restrictions_())),
          g(std::move(minimum_extents_())),
          require_invertible_()};
}

inline bool Restrictions::require_invertible_() const { return req_invertible_; }

inline SmallVector<Restriction>& Restrictions::dimension_restrictions_()
{
  return dim_restrictions_;
}

inline const SmallVector<Restriction>& Restrictions::dimension_restrictions_() const
{
  return dim_restrictions_;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM>& Restrictions::minimum_extents_()
{
  return min_extents_;
}

inline const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& Restrictions::minimum_extents_() const
{
  return min_extents_;
}

}  // namespace legate::detail
