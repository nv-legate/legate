/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/redop/detail/atomic_wrapper.h>
#include <legate/utilities/detail/type_traits.h>

#include <cuda/std/atomic>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

namespace legate::detail {

template <typename F>
template <typename T>
LEGATE_HOST_DEVICE void AtomicWrapper<F>::operator()(T& lhs,
                                                     T rhs,
                                                     ::cuda::std::memory_order order) const noexcept
{
  auto ref = ::cuda::std::atomic_ref<T>{lhs};

  if constexpr (is_instance_of_v<F, ::cuda::std::plus> && ::cuda::std::is_arithmetic_v<T>) {
    ref.fetch_add(rhs, order);
  } else if constexpr (is_instance_of_v<F, ::cuda::std::minus> && ::cuda::std::is_arithmetic_v<T>) {
    ref.fetch_sub(rhs, order);
  } else if constexpr (is_instance_of_v<F, ::cuda::std::bit_and> && ::cuda::std::is_integral_v<T>) {
    ref.fetch_and(rhs, order);
  } else if constexpr (is_instance_of_v<F, ::cuda::std::bit_or> && ::cuda::std::is_integral_v<T>) {
    ref.fetch_or(rhs, order);
  } else if constexpr (is_instance_of_v<F, ::cuda::std::bit_xor> && ::cuda::std::is_integral_v<T>) {
    ref.fetch_xor(rhs, order);
  } else {
    auto oldval = ref.load(::cuda::std::memory_order_relaxed);
    T newval;

    do {
      newval = F{}(oldval, rhs);
    } while (!ref.compare_exchange_weak(oldval,
                                        newval,
                                        /* success */ order,
                                        /* failure */ ::cuda::std::memory_order_relaxed));
  }
}

}  // namespace legate::detail
