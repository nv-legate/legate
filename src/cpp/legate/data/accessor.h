/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/typedefs.h>

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

  // `origin` points to the element at `lo`; `strides` are in elements.
  LEGATE_HOST_DEVICE AccessorBase(ELEM* origin,
                                  const legate::Point<DIM>& lo,
                                  const std::size_t strides[DIM])
    : base_{origin}, lo_{lo}
  {
    for (int i = 0; i < DIM; ++i) {
      strides_[i] = strides[i];
    }
  }

  // Element offset (in elements) of `p` relative to the accessor's origin.
  LEGATE_HOST_DEVICE std::int64_t offset(const legate::Point<DIM>& p) const
  {
    std::int64_t off = 0;
    for (int i = 0; i < DIM; ++i) {
      off += static_cast<std::int64_t>(p[i] - lo_[i]) * static_cast<std::int64_t>(strides_[i]);
    }
    return off;
  }

  LEGATE_HOST_DEVICE ELEM* ptr(const legate::Point<DIM>& p) const { return base_ + offset(p); }

  LEGATE_HOST_DEVICE ELEM* ptr(const legate::Rect<DIM>& r) const { return ptr(r.lo); }

  LEGATE_HOST_DEVICE ELEM* ptr(const legate::Rect<DIM>& r, std::size_t strides[DIM]) const
  {
    for (int i = 0; i < DIM; ++i) {
      strides[i] = strides_[i];
    }
    return ptr(r.lo);
  }

  // True when every stride is zero, i.e. the store is a broadcast scalar.
  LEGATE_HOST_DEVICE bool is_scalar() const
  {
    for (int i = 0; i < DIM; ++i) {
      if (strides_[i] != 0) {
        return false;
      }
    }
    return true;
  }

  // Dense-layout predicates, computed from the strides (mirroring the semantics
  // of Realm's AffineAccessor::is_dense_*). A dimension of extent 1 imposes no
  // constraint on its stride.
  LEGATE_HOST_DEVICE bool is_dense_row_major(const legate::Rect<DIM>& r) const
  {
    std::size_t expected = 1;
    for (int i = DIM - 1; i >= 0; --i) {
      const std::size_t extent = static_cast<std::size_t>(r.hi[i] - r.lo[i] + 1);
      if (extent > 1 && strides_[i] != expected) {
        return false;
      }
      expected *= extent;
    }
    return true;
  }

  LEGATE_HOST_DEVICE bool is_dense_col_major(const legate::Rect<DIM>& r) const
  {
    std::size_t expected = 1;
    for (int i = 0; i < DIM; ++i) {
      const std::size_t extent = static_cast<std::size_t>(r.hi[i] - r.lo[i] + 1);
      if (extent > 1 && strides_[i] != expected) {
        return false;
      }
      expected *= extent;
    }
    return true;
  }

  LEGATE_HOST_DEVICE bool is_dense_arbitrary(const legate::Rect<DIM>& r) const
  {
    // Collect (stride, extent) for the dimensions that actually vary, sort by
    // stride ascending, and check they pack contiguously from stride 1.
    std::size_t strides[DIM];
    std::size_t extents[DIM];
    int n = 0;
    for (int i = 0; i < DIM; ++i) {
      const std::size_t extent = static_cast<std::size_t>(r.hi[i] - r.lo[i] + 1);
      if (extent > 1) {
        strides[n] = strides_[i];
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

 protected:
  ELEM* base_{nullptr};
  legate::Point<DIM> lo_{};
  std::size_t strides_[DIM]{};
};

}  // namespace detail

// --- Read-only ------------------------------------------------------------
template <typename T, int DIM>
class AccessorRO : public detail::AccessorBase<const T, DIM> {
 public:
  using detail::AccessorBase<const T, DIM>::AccessorBase;
  using detail::AccessorBase<const T, DIM>::ptr;

  LEGATE_HOST_DEVICE const T& operator[](const legate::Point<DIM>& p) const
  {
    return *this->ptr(p);
  }
};

// --- Write-only -----------------------------------------------------------
template <typename T, int DIM>
class AccessorWO : public detail::AccessorBase<T, DIM> {
 public:
  using detail::AccessorBase<T, DIM>::AccessorBase;
  using detail::AccessorBase<T, DIM>::ptr;

  LEGATE_HOST_DEVICE T& operator[](const legate::Point<DIM>& p) const { return *this->ptr(p); }
};

// --- Read-write -----------------------------------------------------------
template <typename T, int DIM>
class AccessorRW : public detail::AccessorBase<T, DIM> {
 public:
  using detail::AccessorBase<T, DIM>::AccessorBase;
  using detail::AccessorBase<T, DIM>::ptr;

  LEGATE_HOST_DEVICE T& operator[](const legate::Point<DIM>& p) const { return *this->ptr(p); }
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

  LEGATE_HOST_DEVICE void reduce(const legate::Point<DIM>& p, RHS value) const
  {
    REDOP::template apply<EXCLUSIVE>(*this->ptr(p), value);
  }

  // Proxy so that `acc[p] <<= value` performs the reduction.
  class ReductionProxy {
   public:
    LEGATE_HOST_DEVICE ReductionProxy(RHS* target) : target_{target} {}

    LEGATE_HOST_DEVICE void operator<<=(RHS value) const { reduce(value); }

    LEGATE_HOST_DEVICE void reduce(RHS value) const
    {
      REDOP::template apply<EXCLUSIVE>(*target_, value);
    }

   private:
    RHS* target_;
  };

  LEGATE_HOST_DEVICE ReductionProxy operator[](const legate::Point<DIM>& p) const
  {
    return ReductionProxy{this->ptr(p)};
  }
};

}  // namespace legate
