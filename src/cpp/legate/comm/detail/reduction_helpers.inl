/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/abort.h>
#include <legate/utilities/detail/traced_exception.h>

#include <algorithm>
#include <functional>
#include <type_traits>

namespace legate::detail::comm::coll {

template <typename T>
void apply_reduction_typed(void* dst, const void* src, unsigned count, ReductionOpKind op)
{
  auto* dst_typed       = static_cast<T*>(dst);
  const auto* src_typed = static_cast<const T*>(src);

  switch (op) {
    case legate::ReductionOpKind::ADD: {
      std::transform(dst_typed, dst_typed + count, src_typed, dst_typed, std::plus<T>{});
      break;
    }
    case legate::ReductionOpKind::MUL: {
      std::transform(dst_typed, dst_typed + count, src_typed, dst_typed, std::multiplies<T>{});
      break;
    }
    case legate::ReductionOpKind::MAX: {
      std::transform(dst_typed, dst_typed + count, src_typed, dst_typed, [](T a, T b) {
        return std::max(a, b);
      });
      break;
    }
    case legate::ReductionOpKind::MIN: {
      std::transform(dst_typed, dst_typed + count, src_typed, dst_typed, [](T a, T b) {
        return std::min(a, b);
      });
      break;
    }
    case legate::ReductionOpKind::AND: {
      if constexpr (std::is_integral_v<T>) {
        std::transform(dst_typed, dst_typed + count, src_typed, dst_typed, std::bit_and<T>{});
      } else {
        throw legate::detail::TracedException<std::invalid_argument>{
          "Reduction does not support non-integral types with AND"};
      }
      break;
    }
    case legate::ReductionOpKind::OR: {
      if constexpr (std::is_integral_v<T>) {
        std::transform(dst_typed, dst_typed + count, src_typed, dst_typed, std::bit_or<T>{});
      } else {
        throw legate::detail::TracedException<std::invalid_argument>{
          "Reduction does not support non-integral types with OR"};
      }
      break;
    }
    case legate::ReductionOpKind::XOR: {
      if constexpr (std::is_integral_v<T>) {
        std::transform(dst_typed, dst_typed + count, src_typed, dst_typed, std::bit_xor<T>{});
      } else {
        throw legate::detail::TracedException<std::invalid_argument>{
          "Reduction does not support non-integral types with XOR"};
      }
      break;
    }
  }
}

}  // namespace legate::detail::comm::coll
