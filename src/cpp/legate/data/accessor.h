/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/geometry/point.h>
#include <legate/geometry/rect.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>

namespace legate {

namespace detail {

// Shared base pointer + element-stride storage and index math for the read /
// write / read-write store accessors. `ELEM` is `T` or `const T` to encode the
// privilege in the returned references.
template <typename ELEM, int DIM>
class AccessorBase {
 public:
  AccessorBase() = default;

  // `base` points to the element at the store's coordinate origin (the point
  // `(0, ..., 0)`); `strides` are in elements. The base pointer is already
  // translated to the origin at store-creation time, so `offset()` is a plain
  // dot product with no per-call subtraction of `lo`.
  LEGATE_HOST_DEVICE AccessorBase(ELEM* base, legate::Point<DIM> strides);

  // Element offset (in elements) of `p` relative to the coordinate origin.
  LEGATE_HOST_DEVICE std::int64_t offset(const legate::Point<DIM>& p) const;

  LEGATE_HOST_DEVICE ELEM* ptr(const legate::Point<DIM>& p) const;

  LEGATE_HOST_DEVICE ELEM* ptr(const legate::Rect<DIM>& r) const;

  LEGATE_HOST_DEVICE ELEM* ptr(const legate::Rect<DIM>& r, std::size_t strides[DIM]) const;

  // True when every stride is zero, i.e. the store is a broadcast scalar.
  LEGATE_HOST_DEVICE bool is_scalar() const;

  // Dense-layout predicates, computed from the strides (mirroring the semantics
  // of Realm's AffineAccessor::is_dense_*). A dimension of extent 1 imposes no
  // constraint on its stride.
  LEGATE_HOST_DEVICE bool is_dense_row_major(const legate::Rect<DIM>& r) const;

  LEGATE_HOST_DEVICE bool is_dense_col_major(const legate::Rect<DIM>& r) const;

  LEGATE_HOST_DEVICE bool is_dense_arbitrary(const legate::Rect<DIM>& r) const;

 protected:
  ELEM* base_{nullptr};
  legate::Point<DIM> strides_{};
};

}  // namespace detail

// --- Read-only ------------------------------------------------------------
template <typename T, int DIM>
class AccessorRO : public detail::AccessorBase<const T, DIM> {
 public:
  using detail::AccessorBase<const T, DIM>::AccessorBase;
  using detail::AccessorBase<const T, DIM>::ptr;

  LEGATE_HOST_DEVICE const T& operator[](const legate::Point<DIM>& p) const;
};

// --- Write-only -----------------------------------------------------------
template <typename T, int DIM>
class AccessorWO : public detail::AccessorBase<T, DIM> {
 public:
  using detail::AccessorBase<T, DIM>::AccessorBase;
  using detail::AccessorBase<T, DIM>::ptr;

  LEGATE_HOST_DEVICE T& operator[](const legate::Point<DIM>& p) const;
};

// --- Read-write -----------------------------------------------------------
template <typename T, int DIM>
class AccessorRW : public detail::AccessorBase<T, DIM> {
 public:
  using detail::AccessorBase<T, DIM>::AccessorBase;
  using detail::AccessorBase<T, DIM>::ptr;

  LEGATE_HOST_DEVICE T& operator[](const legate::Point<DIM>& p) const;
};

// --- Reduction ------------------------------------------------------------
// Mirrors Legion's ReductionAccessor: both `reduce(p, v)` and `acc[p] <<= v`
// fold `v` into the target element via `REDOP::apply<EXCLUSIVE>`.
template <typename REDOP, bool EXCLUSIVE, int DIM>
class AccessorRD : public detail::AccessorBase<typename REDOP::RHS, DIM> {
 public:
  using RHS  = typename REDOP::RHS;
  using Base = detail::AccessorBase<RHS, DIM>;
  using Base::Base;
  using Base::ptr;

  LEGATE_HOST_DEVICE void reduce(const legate::Point<DIM>& p, RHS value) const;

  // Proxy so that `acc[p] <<= value` performs the reduction.
  class ReductionProxy {
   public:
    LEGATE_HOST_DEVICE ReductionProxy(RHS* target);

    LEGATE_HOST_DEVICE void operator<<=(RHS value) const;

    LEGATE_HOST_DEVICE void reduce(RHS value) const;

   private:
    RHS* target_;
  };

  LEGATE_HOST_DEVICE ReductionProxy operator[](const legate::Point<DIM>& p) const;
};

}  // namespace legate

#include <legate/data/accessor.inl>
