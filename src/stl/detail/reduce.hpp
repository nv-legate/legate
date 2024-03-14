/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/utilities/assert.h"

#include "elementwise.hpp"
#include "launch_task.hpp"
#include "store.hpp"

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

namespace detail {

////////////////////////////////////////////////////////////////////////////////////////////////////
template <auto Identity, typename Apply, typename Fold = Apply>
class basic_reduction {
 public:
  using reduction_type                 = basic_reduction;
  using value_type                     = std::remove_cv_t<decltype(Identity)>;
  using RHS                            = value_type;
  using LHS                            = value_type;
  static constexpr value_type identity = Identity;

  template <bool Exclusive>
  LEGATE_HOST_DEVICE static void apply(LHS& lhs, RHS rhs)
  {
    // TODO(ericnieblier): use atomic operations when Exclusive is false
    lhs = Apply()(lhs, rhs);
  }

  template <bool Exclusive>
  LEGATE_HOST_DEVICE static void fold(RHS& lhs, RHS rhs)
  {
    // TODO(ericniebler): use atomic operations when Exclusive is false
    lhs = Fold()(lhs, rhs);
  }

  void operator()(RHS& lhs, RHS rhs) const
  {
    // TODO(ericniebler): how to support atomic operations here?
    this->fold<true>(lhs, rhs);
  }
};

// The legate.stl library's `reduce` function wants reductions to also define a
// function-call operator that knows how to apply the reduction to the range's
// value-type, e.g., to apply it elementwise to all the elements of an mdspan.
template <typename Reduction>
class reduction_wrapper : public Reduction {
 public:
  using reduction_type = Reduction;

  template <typename LHS, typename RHS>
  void operator()(LHS&& lhs, RHS rhs) const
  {
    std::forward<LHS>(lhs) <<= rhs;
  }

  template <typename LHS, typename RHS>
  LEGATE_HOST_DEVICE void operator()(std::size_t tid, LHS&& lhs, RHS rhs) const
  {
    if (tid == 0) {
      std::forward<LHS>(lhs) <<= rhs;
    }
  }
};

template <typename Reduction>
reduction_wrapper(Reduction) -> reduction_wrapper<Reduction>;

template <typename Reduction>
class elementwise_reduction : public Reduction {
 public:
  using reduction_type = Reduction;

  // This function expects to be passed mdspan objects
  template <typename State, typename Value>
  void operator()(State state, Value value) const
  {
    LegateAssert(state.extents() == value.extents());

    const std::size_t size = state.size();

    const auto lhs_ptr = state.data_handle();
    const auto rhs_ptr = value.data_handle();

    const auto& lhs_map = state.mapping();
    const auto& rhs_map = value.mapping();

    const auto& lhs_acc = state.accessor();
    const auto& rhs_acc = value.accessor();

    for (std::size_t idx = 0; idx < size; ++idx) {
      auto&& lhs = lhs_acc.access(lhs_ptr, lhs_map(idx));
      auto&& rhs = rhs_acc.access(rhs_ptr, rhs_map(idx));

      lhs <<= rhs;  // reduce
    }
  }

  // This function expects to be passed mdspan objects. This
  // is the GPU implementation, where idx is the thread id.
  template <typename State, typename Value>
  LEGATE_HOST_DEVICE void operator()(std::size_t tid, State state, Value value) const
  {
    LegateAssert(state.extents() == value.extents());

    const std::size_t size = state.size();

    const std::size_t idx = tid;
    if (idx >= size) {
      return;
    }

    const auto lhs_ptr = state.data_handle();
    const auto rhs_ptr = value.data_handle();

    const auto& lhs_map = state.mapping();
    const auto& rhs_map = value.mapping();

    const auto& lhs_acc = state.accessor();
    const auto& rhs_acc = value.accessor();

    auto&& lhs = lhs_acc.access(lhs_ptr, lhs_map(idx));
    auto&& rhs = rhs_acc.access(rhs_ptr, rhs_map(idx));

    lhs <<= rhs;  // reduce
  }
};

template <typename Reduction>
elementwise_reduction(Reduction) -> elementwise_reduction<Reduction>;

template <typename Reduction>
elementwise_reduction(reduction_wrapper<Reduction>) -> elementwise_reduction<Reduction>;

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ValueType, ValueType Identity, typename Apply, typename Fold = Apply>
detail::basic_reduction<Identity, Apply, Fold> make_reduction(Apply, Fold = {})
{
  static_assert(legate::type_code_of<ValueType> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  static_assert(std::is_invocable_r_v<ValueType, Apply, ValueType, ValueType>,
                "The apply function must be callable with two arguments of type ValueType "
                "and must return a value of type ValueType");
  static_assert(std::is_invocable_r_v<ValueType, Fold, ValueType, ValueType>,
                "The fold function must be callable with two arguments of type ValueType "
                "and must return a value of type ValueType");
  static_assert(std::is_empty_v<Apply>, "The apply function must be stateless");
  static_assert(std::is_empty_v<Fold>, "The fold function must be stateless");
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <auto Identity, typename Apply, typename Fold = Apply>
detail::basic_reduction<Identity, Apply, Fold> make_reduction(Apply, Fold = {})
{
  using value_type = std::remove_cv_t<decltype(Identity)>;
  return stl::make_reduction<value_type, Identity>(Apply{}, Fold{});
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ValueType, typename Reduction>
  requires(legate_reduction<Reduction>)  //
[[nodiscard]] auto as_reduction(Reduction red)
{
  using RHS = typename Reduction::RHS;
  if constexpr (callable<Reduction, RHS&, RHS>) {
    return red;
  } else {
    return detail::reduction_wrapper{std::move(red)};
  }
}

template <typename ValueType, typename T>
[[nodiscard]] auto as_reduction(std::plus<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::reduction_wrapper{legate::SumReduction<Type>()};
}

template <typename ValueType, typename T>
[[nodiscard]] auto as_reduction(std::minus<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::reduction_wrapper{legate::DiffReduction<Type>()};
}

template <typename ValueType, typename T>
[[nodiscard]] auto as_reduction(std::multiplies<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::reduction_wrapper{legate::ProdReduction<Type>()};
}

template <typename ValueType, typename T>
[[nodiscard]] auto as_reduction(std::divides<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::reduction_wrapper{legate::DivReduction<Type>()};
}

// TODO(ericniebler): min and max reductions

template <typename ValueType, typename T>
[[nodiscard]] auto as_reduction(std::logical_or<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::reduction_wrapper{legate::OrReduction<Type>()};
}

template <typename ValueType, typename T>
[[nodiscard]] auto as_reduction(std::logical_and<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::reduction_wrapper{legate::AndReduction<Type>()};
}

// TODO(ericniebler): logical xor

template <typename ValueType, typename Function>
[[nodiscard]] auto as_reduction(const detail::elementwise<Function>& fn)
{
  return detail::elementwise_reduction{stl::as_reduction<ValueType>(fn.function())};
}

template <typename Fun, typename ValueType>
using as_reduction_t = decltype(stl::as_reduction<ValueType>(std::declval<Fun>()));

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename InputRange, typename Init, typename BinaryOperation>  //
  requires(logical_store_like<InputRange> && logical_store_like<Init> &&
           legate_reduction<as_reduction_t<BinaryOperation, element_type_of_t<Init>>>)  //
[[nodiscard]] auto reduce(InputRange&& input, Init&& init, BinaryOperation op)
  -> logical_store<element_type_of_t<InputRange>, dim_of_v<InputRange> - 1>
{
  detail::check_function_type<as_reduction_t<BinaryOperation, element_type_of_t<Init>>>();
  static_assert(std::is_same_v<value_type_of_t<InputRange>, value_type_of_t<Init>>);
  static_assert(dim_of_v<InputRange> == dim_of_v<Init> + 1);
  static_assert(std::is_empty_v<BinaryOperation>,
                "Only stateless reduction operations are currently supported");

  // promote the initial value to the same shape as the input so they can
  // be aligned
  using Input       = std::decay_t<InputRange>;
  using InputPolicy = typename Input::policy;

  LogicalStore out =
    InputPolicy::aligned_promote(get_logical_store(input), get_logical_store(init));
  LegateAssert(out.dim() == init.dim() + 1);

  using OutputRange = slice_view<value_type_of_t<Input>, dim_of_v<Input>, InputPolicy>;
  OutputRange output{std::move(out)};

  stl::launch_task(
    stl::inputs(std::forward<InputRange>(input)),
    stl::reduction(std::move(output), stl::as_reduction<element_type_of_t<Init>>(std::move(op))),
    stl::constraints(stl::align(stl::reduction, stl::inputs[0])));

  return as_typed<element_type_of_t<Init>, dim_of_v<Init>>(get_logical_store(init));
}

}  // namespace legate::stl

#include "suffix.hpp"
