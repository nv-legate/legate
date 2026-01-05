/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <cuda/std/atomic>

namespace legate::detail {

/**
 * @brief Make a non-atomic binary functor act atomically.
 *
 * @tparam F The non-atomic functor to wrap. For example std::plus<>.
 *
 * The functor must be a default-constructable binary functor which exposes a call operator
 * with the following signature:
 *
 * T operator()(const T&, const T&)
 */
template <typename F>
class AtomicWrapper {
 public:
  /**
   * @brief Invoke the function
   *
   * @param lhs The left argument of the functor. It is updated in-place.
   * @param rhs The right argument of the functor.
   * @param order The memory order in which to perform the operation.
   */
  template <typename T>
  LEGATE_HOST_DEVICE void operator()(
    T& lhs,
    T rhs,
    ::cuda::std::memory_order order = ::cuda::std::memory_order_relaxed) const noexcept;
};

}  // namespace legate::detail

#include <legate/redop/detail/atomic_wrapper.inl>
