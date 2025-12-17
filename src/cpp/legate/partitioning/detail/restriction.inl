/*
 *  * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *   * reserved.
 *    * SPDX-License-Identifier: Apache-2.0
 *     */

#pragma once

#include <legate/partitioning/detail/restriction.h>

namespace legate::detail {

inline Restrictions::Restrictions(std::uint32_t ndim)
  : dim_restrictions_{tags::size_tag, ndim, Restriction::ALLOW}
{
}

inline Restrictions::Restrictions(SmallVector<Restriction> dimension_restrictions,
                                  bool require_invertible)
  : req_invertible_{require_invertible}, dim_restrictions_{std::move(dimension_restrictions)}
{
}

inline void Restrictions::set_require_invertible(bool new_value) { req_invertible_ = new_value; }

template <typename FUNC>
Restrictions Restrictions::map(FUNC&& f) &&
{
  return {f(std::move(dimension_restrictions_())), require_invertible_()};
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

}  // namespace legate::detail
