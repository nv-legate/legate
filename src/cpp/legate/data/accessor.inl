/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate/data/accessor.h>

#include <cstddef>
#include <cstdint>

namespace legate {

namespace detail {

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE AccessorBase<ELEM, DIM>::AccessorBase(ELEM* base, legate::Point<DIM> strides)
  // `base` already points at the coordinate origin, so `offset()` is a plain dot
  // product of the access point with the strides.
  : base_{base}, strides_{std::move(strides)}
{
}

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE std::int64_t AccessorBase<ELEM, DIM>::offset(const legate::Point<DIM>& p) const
{
  return p.dot(strides_);
}

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE ELEM* AccessorBase<ELEM, DIM>::ptr(const legate::Point<DIM>& p) const
{
  return base_ + offset(p);
}

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE ELEM* AccessorBase<ELEM, DIM>::ptr(const legate::Rect<DIM>& r) const
{
  return ptr(r.lo);
}

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE ELEM* AccessorBase<ELEM, DIM>::ptr(const legate::Rect<DIM>& r,
                                                      std::size_t strides[DIM]) const
{
  for (int i = 0; i < DIM; ++i) {
    strides[i] = static_cast<std::size_t>(strides_[i]);
  }
  return ptr(r.lo);
}

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE bool AccessorBase<ELEM, DIM>::is_scalar() const
{
  for (int i = 0; i < DIM; ++i) {
    if (strides_[i] != 0) {
      return false;
    }
  }
  return true;
}

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE bool AccessorBase<ELEM, DIM>::is_dense_row_major(
  const legate::Rect<DIM>& r) const
{
  std::size_t expected = 1;
  for (int i = DIM - 1; i >= 0; --i) {
    const std::size_t extent = static_cast<std::size_t>(r.hi[i] - r.lo[i] + 1);
    if (extent > 1 && static_cast<std::size_t>(strides_[i]) != expected) {
      return false;
    }
    expected *= extent;
  }
  return true;
}

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE bool AccessorBase<ELEM, DIM>::is_dense_col_major(
  const legate::Rect<DIM>& r) const
{
  std::size_t expected = 1;
  for (int i = 0; i < DIM; ++i) {
    const std::size_t extent = static_cast<std::size_t>(r.hi[i] - r.lo[i] + 1);
    if (extent > 1 && static_cast<std::size_t>(strides_[i]) != expected) {
      return false;
    }
    expected *= extent;
  }
  return true;
}

template <typename ELEM, int DIM>
LEGATE_HOST_DEVICE bool AccessorBase<ELEM, DIM>::is_dense_arbitrary(
  const legate::Rect<DIM>& r) const
{
  // Collect (stride, extent) for the dimensions that actually vary, sort by
  // stride ascending, and check they pack contiguously from stride 1.
  std::size_t strides[DIM];
  std::size_t extents[DIM];
  int n = 0;
  for (int i = 0; i < DIM; ++i) {
    const std::size_t extent = static_cast<std::size_t>(r.hi[i] - r.lo[i] + 1);
    if (extent > 1) {
      strides[n] = static_cast<std::size_t>(strides_[i]);
      extents[n] = extent;
      ++n;
    }
  }
  for (int a = 1; a < n; ++a) {
    const std::size_t sk = strides[a];
    const std::size_t ek = extents[a];
    int b                = a - 1;
    while (b >= 0 && strides[b] > sk) {
      strides[b + 1] = strides[b];
      extents[b + 1] = extents[b];
      --b;
    }
    strides[b + 1] = sk;
    extents[b + 1] = ek;
  }
  std::size_t expected = 1;
  for (int a = 0; a < n; ++a) {
    if (strides[a] != expected) {
      return false;
    }
    expected *= extents[a];
  }
  return true;
}

}  // namespace detail

template <typename T, int DIM>
LEGATE_HOST_DEVICE const T& AccessorRO<T, DIM>::operator[](const legate::Point<DIM>& p) const
{
  return *this->ptr(p);
}

template <typename T, int DIM>
LEGATE_HOST_DEVICE T& AccessorWO<T, DIM>::operator[](const legate::Point<DIM>& p) const
{
  return *this->ptr(p);
}

template <typename T, int DIM>
LEGATE_HOST_DEVICE T& AccessorRW<T, DIM>::operator[](const legate::Point<DIM>& p) const
{
  return *this->ptr(p);
}

template <typename REDOP, bool EXCLUSIVE, int DIM>
LEGATE_HOST_DEVICE void AccessorRD<REDOP, EXCLUSIVE, DIM>::reduce(const legate::Point<DIM>& p,
                                                                  RHS value) const
{
  REDOP::template apply<EXCLUSIVE>(*this->ptr(p), value);
}

template <typename REDOP, bool EXCLUSIVE, int DIM>
LEGATE_HOST_DEVICE AccessorRD<REDOP, EXCLUSIVE, DIM>::ReductionProxy::ReductionProxy(RHS* target)
  : target_{target}
{
}

template <typename REDOP, bool EXCLUSIVE, int DIM>
LEGATE_HOST_DEVICE void AccessorRD<REDOP, EXCLUSIVE, DIM>::ReductionProxy::operator<<=(
  RHS value) const
{
  reduce(value);
}

template <typename REDOP, bool EXCLUSIVE, int DIM>
LEGATE_HOST_DEVICE void AccessorRD<REDOP, EXCLUSIVE, DIM>::ReductionProxy::reduce(RHS value) const
{
  REDOP::template apply<EXCLUSIVE>(*target_, value);
}

template <typename REDOP, bool EXCLUSIVE, int DIM>
LEGATE_HOST_DEVICE typename AccessorRD<REDOP, EXCLUSIVE, DIM>::ReductionProxy
AccessorRD<REDOP, EXCLUSIVE, DIM>::operator[](const legate::Point<DIM>& p) const
{
  return ReductionProxy{this->ptr(p)};
}

}  // namespace legate
