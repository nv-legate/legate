/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/partitioning/detail/partitioning_tasks.h>

#include <legion/api/redop.h>

namespace legate::detail {

// Copied and modified from Legion
template <typename OP, typename T>
#if LEGATE_DEFINED(LEGATE_DEVICE_COMPILE)
LEGATE_DEVICE
#endif
  void wrap_with_cas(OP op, T& lhs, T rhs)
{
#if LEGATE_DEFINED(LEGATE_DEVICE_COMPILE)
  T newval = lhs, oldval;
  // atomicCAS has no override for std::uint64_t, so we suppress the clang tidy error
  auto* ptr = reinterpret_cast<unsigned long long int*>(&lhs);  // NOLINT(google-runtime-int)
  do {
    oldval = newval;
    newval = op(newval, rhs);
    newval = Legion::__ulonglong_as_longlong(atomicCAS(
      ptr, Legion::__longlong_as_ulonglong(oldval), Legion::__longlong_as_ulonglong(newval)));
  } while (oldval != newval);
#elif defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
  std::atomic_ref<T> atomic{lhs};
  auto oldval = atomic.load();
  auto newval;
  do {
    newval = op(oldval, rhs);
  } while (!atomic.compare_exchange_weak(oldval, newval));
#else
  T oldval, newval;
  const Legion::TypePunning::Pointer<T> pointer{static_cast<void*>(&lhs)};
  do {
    oldval = *pointer;
    newval = op(oldval, rhs);
  } while (!__sync_bool_compare_and_swap(static_cast<T*>(pointer), oldval, newval));
#endif
}

template <std::int32_t NDIM>
template <bool EXCLUSIVE>
LEGATE_HOST_DEVICE inline /*static*/ void ElementWiseMax<NDIM>::apply(LHS& lhs, RHS rhs)
{
  static_assert(sizeof(coord_t) == sizeof(std::uint64_t), "coord_t has an unexpected size");

  if constexpr (EXCLUSIVE) {
    for (std::int32_t dim = 0; dim < NDIM; ++dim) {
      lhs[dim] = std::max(lhs[dim], rhs[dim]);
    }
  } else {
    for (std::int32_t dim = 0; dim < NDIM; ++dim) {
      wrap_with_cas([](auto a, auto b) constexpr { return std::max(a, b); }, lhs[dim], rhs[dim]);
    }
  }
}

template <std::int32_t NDIM>
template <bool EXCLUSIVE>
LEGATE_HOST_DEVICE inline /*static*/ void ElementWiseMax<NDIM>::fold(RHS& rhs1, RHS rhs2)
{
  apply<EXCLUSIVE>(rhs1, rhs2);
}

template <std::int32_t NDIM>
template <bool EXCLUSIVE>
LEGATE_HOST_DEVICE inline /*static*/ void ElementWiseMin<NDIM>::apply(LHS& lhs, RHS rhs)
{
  static_assert(sizeof(coord_t) == sizeof(std::uint64_t), "coord_t has an unexpected size");

  if constexpr (EXCLUSIVE) {
    for (std::int32_t dim = 0; dim < NDIM; ++dim) {
      lhs[dim] = std::min(lhs[dim], rhs[dim]);
    }
  } else {
    for (std::int32_t dim = 0; dim < NDIM; ++dim) {
      wrap_with_cas([](auto a, auto b) constexpr { return std::min(a, b); }, lhs[dim], rhs[dim]);
    }
  }
}

template <std::int32_t NDIM>
template <bool EXCLUSIVE>
LEGATE_HOST_DEVICE inline /*static*/ void ElementWiseMin<NDIM>::fold(RHS& rhs1, RHS rhs2)
{
  apply<EXCLUSIVE>(rhs1, rhs2);
}

}  // namespace legate::detail
