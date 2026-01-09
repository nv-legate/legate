/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/redop/detail/base.h>
#include <legate/utilities/macros.h>

#include <type_traits>
#include <utility>

namespace legate::detail {

template <typename T, ReductionOpKind K, template <typename> typename A>
template <bool EXCLUSIVE>
LEGATE_HOST_DEVICE /*static*/ void BaseReduction<T, K, A>::apply(LHS& lhs, RHS rhs)
{
  if constexpr (EXCLUSIVE) {
    // We assign the result of the op to lhs and rhs1 respectively, so the reduction operator
    // must return the value (and must not, for example, return void)
    static_assert(std::is_invocable_r_v<LHS, redop_type, LHS&, RHS>);

    lhs = redop_type{}(lhs, std::move(rhs));
  } else {
    // As opposed to the non-atomic operators, the atomic redops must act only on their
    // arguments, so should return void.
    static_assert(std::is_invocable_r_v<void, atomic_redop_type, LHS&, RHS>);

    atomic_redop_type{}(lhs, std::move(rhs));
  }
}

template <typename T, ReductionOpKind K, template <typename> typename A>
template <bool EXCLUSIVE>
LEGATE_HOST_DEVICE /*static*/ void BaseReduction<T, K, A>::fold(RHS& rhs1, RHS rhs2)
{
  // We use the same implementation for apply and fold, so these need to be the same. The
  // reason we can use the same implementation is because we don't support subtraction or
  // division reduction operators.
  static_assert(std::is_same_v<LHS, RHS>);
  apply<EXCLUSIVE>(rhs1, std::move(rhs2));
}

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
template <typename T, ReductionOpKind K, template <typename> typename A>
template <bool EXCLUSIVE>
LEGATE_HOST_DEVICE /*static*/ void BaseReduction<T, K, A>::apply_cuda(LHS& lhs, RHS rhs)
{
  apply<EXCLUSIVE>(lhs, std::move(rhs));
}

template <typename T, ReductionOpKind K, template <typename> typename A>
template <bool EXCLUSIVE>
LEGATE_HOST_DEVICE /*static*/ void BaseReduction<T, K, A>::fold_cuda(RHS& rhs1, RHS rhs2)
{
  fold<EXCLUSIVE>(rhs1, std::move(rhs2));
}
#endif

}  // namespace legate::detail
