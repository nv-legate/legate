/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/redop/detail/atomic_wrapper.h>
#include <legate/type/type_traits.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/macros.h>

#include <legion/api/config.h>

#include <cuda/std/functional>

#include <type_traits>

namespace legate::detail {

/**
 * @brief Maps a ReductionOpKind to its corresponding CUDA functor type.
 *
 * Provides a type alias `type` for the functor implementing the reduction
 * operation. Unsupported kinds trigger a static assertion.
 *
 * @tparam KIND The reduction operation kind.
 */
template <ReductionOpKind KIND>
struct redop_kind_op {  // NOLINT(readability-identifier-naming)
  // Do this static assert so make the error message nicer
  static_assert(sizeof(KIND) != 0, "Unhandled reduction operation");
};

template <>
struct redop_kind_op<ReductionOpKind::ADD> {
  using type = ::cuda::std::plus<>;
};

template <>
struct redop_kind_op<ReductionOpKind::MUL> {
  using type = ::cuda::std::multiplies<>;
};

template <>
struct redop_kind_op<ReductionOpKind::MAX> {
 private:
  // Exists because cuda::std::maximum() doesn't exist, and we don't want to pull in thrust for
  // it.
  class Maximum {
   public:
    template <typename T1, typename T2>
    [[nodiscard]] constexpr std::common_type_t<T1, T2> operator()(const T1& lhs,
                                                                  const T2& rhs) const
      noexcept(noexcept((lhs < rhs) ? rhs : lhs))
    {
      return (lhs < rhs) ? rhs : lhs;
    }
  };

 public:
  using type = Maximum;
};

template <>
struct redop_kind_op<ReductionOpKind::MIN> {
 private:
  // Exists because cuda::std::minimum() doesn't exist, and we don't want to pull in thrust for
  // it.
  class Minimum {
   public:
    template <typename T1, typename T2>
    [[nodiscard]] constexpr std::common_type_t<T1, T2> operator()(const T1& lhs,
                                                                  const T2& rhs) const
      noexcept(noexcept((lhs < rhs) ? lhs : rhs))
    {
      return (lhs < rhs) ? lhs : rhs;
    }
  };

 public:
  using type = Minimum;
};

template <>
struct redop_kind_op<ReductionOpKind::OR> {
  using type = ::cuda::std::bit_or<>;
};

template <>
struct redop_kind_op<ReductionOpKind::AND> {
  using type = ::cuda::std::bit_and<>;
};

template <>
struct redop_kind_op<ReductionOpKind::XOR> {
  using type = ::cuda::std::bit_xor<>;
};

template <ReductionOpKind KIND>
using redop_kind_op_t = typename redop_kind_op<KIND>::type;

/**
 * @brief Computes the GlobalRedopID for a given reduction operation and type.
 *
 * Maps a ReductionOpKind and a Legate type code to the corresponding Legion built-in reduction
 * operator ID. Special-cases boolean AND as multiplication to match Legion's registration.
 *
 * @param op The reduction operation kind.
 * @param type_code The Legate type code.
 *
 * @return The corresponding GlobalRedopID.
 */
[[nodiscard]] constexpr GlobalRedopID builtin_redop_id(ReductionOpKind op,
                                                       legate::Type::Code type_code)
{
  // FIXME(wonchanl): It's beyond my comprehension why this issue hasn't been triggered by any of
  // our tests until now, cause these reduction op IDs haven't changed since the beginning. In the
  // long run, we should register these built-in operators ourselves in Legate, instead of relying
  // on an equation that is loosely shared by Legate and Legion.
  //
  // We need to special-case the logical-AND reduction for booleans, as it is registered as a
  // prod reduction on the Legion side...
  const auto trans_op = (type_code == legate::Type::Code::BOOL && op == ReductionOpKind::AND)
                          ? ReductionOpKind::MUL
                          : op;

  return static_cast<GlobalRedopID>(
    LEGION_REDOP_BASE + (to_underlying(trans_op) * LEGION_TYPE_TOTAL) + to_underlying(type_code));
}

/**
 * @brief Base class for defining reduction operators.
 *
 * Redops should derive from this class and fill out the various template parameters. The main
 * idea is that at the end of the day, each redop has:
 * ```
 * typename RHS;
 * typename LHS;
 * int REDOP_ID;
 *
 * template <bool EXCLUSIVE>
 * void apply(LHS &, RHS);
 *
 * template <bool EXCLUSIVE>
 * void fold(RHS &, RHS);
 * ```
 *
 * If the build is CUDA enabled, it will also have `apply_cuda()` and `fold_cuda()`.
 *
 * Most types will not have to define a special atomic wrapper, and therefore won't need to
 * pass the final template argument. For example, to declare sum for `std::int32`:
 * ```
 * class SumInt32 : public BaseReduction<std::int32_t, ReductionOpKind::ADD> {};
 * ```
 *
 * @tparam T Base value type (e.g., int, double, complex<float>).
 * @tparam KIND Reduction operation kind. The actual operator functor is determined based on
 * this
 * @tparam AtomicRedopT A templated functor object, which exposes an `operator()()` that does the
 * atomic operation. The template parameter for AtomicRedopT will be the non-atomic operator
 * deduced from `KIND`.
 */
template <typename T,
          ReductionOpKind KIND,
          template <typename> typename AtomicRedopT = AtomicWrapper>
class BaseReduction {
 public:
  using value_type        = T;
  using redop_type        = redop_kind_op_t<KIND>;
  using atomic_redop_type = AtomicRedopT<redop_type>;

  using LHS = value_type;
  using RHS = LHS;

  static constexpr auto CODE       = type_code_of_v<T>;
  static constexpr auto REDOP_KIND = KIND;
  static constexpr auto REDOP_ID   = builtin_redop_id(KIND, CODE);

  /**
   * @brief Applies the reduction to lhs with rhs.
   *
   * @tparam EXCLUSIVE True if the operation is exclusive (i.e. does not need atomics).
   *
   * @param lhs Left-hand side to update.
   * @param rhs Right-hand side value.
   */
  template <bool EXCLUSIVE>
  LEGATE_HOST_DEVICE static void apply(LHS& lhs, RHS rhs);

  /**
   * @brief Left-folds rhs2 into rhs1.
   *
   * @tparam EXCLUSIVE True if the operation is exclusive (i.e. does not need atomics).
   *
   * @param rhs1 Accumulator to update.
   * @param rhs2 Value to fold in.
   */
  template <bool EXCLUSIVE>
  LEGATE_HOST_DEVICE static void fold(RHS& rhs1, RHS rhs2);

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  /**
   * @brief CUDA version of apply().
   */
  template <bool EXCLUSIVE>
  LEGATE_HOST_DEVICE static void apply_cuda(LHS& lhs, RHS rhs);

  /**
   * @brief CUDA version of fold().
   */
  template <bool EXCLUSIVE>
  LEGATE_HOST_DEVICE static void fold_cuda(RHS& rhs1, RHS rhs2);
#endif
};

}  // namespace legate::detail

#include <legate/redop/detail/base.inl>
