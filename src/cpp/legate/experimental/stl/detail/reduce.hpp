/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/elementwise.hpp>
#include <legate/experimental/stl/detail/launch_task.hpp>
#include <legate/experimental/stl/detail/store.hpp>
#include <legate/utilities/assert.h>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

namespace detail {

////////////////////////////////////////////////////////////////////////////////////////////////////
template <auto Identity, typename Apply, typename Fold = Apply>
class BasicReduction {
 public:
  using reduction_type                 = BasicReduction;
  using value_type                     = std::remove_cv_t<decltype(Identity)>;
  using RHS                            = value_type;
  using LHS                            = value_type;
  static constexpr value_type identity = Identity;  // NOLINT(readability-identifier-naming)

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

  void operator()(RHS& lhs, RHS rhs) const;

  // template <class ReductionHelper>
  // void operator()(ReductionHelper&& lhs, RHS rhs) const
  // {
  //   // TODO(ericniebler): how to support atomic operations here?
  //   lhs.reduce(rhs);
  //   //std::forward<ReductionHelper>(lhs) <<= rhs;
  //   //this->fold<true>(lhs, rhs);
  // }
};

// The legate.stl library's `reduce` function wants reductions to also define a
// function-call operator that knows how to apply the reduction to the range's
// value-type, e.g., to apply it elementwise to all the elements of an mdspan.
template <typename Reduction>
class ReductionWrapper : public Reduction {
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
ReductionWrapper(Reduction) -> ReductionWrapper<Reduction>;

template <typename Reduction>
class ElementwiseReduction : public Reduction {
 public:
  using reduction_type = Reduction;

  // This function expects to be passed mdspan objects
  template <typename State, typename Value>
  void operator()(State&& state, Value&& value) const
  {
    LEGATE_ASSERT(state.extents() == value.extents());

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
    LEGATE_ASSERT(state.extents() == value.extents());

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
ElementwiseReduction(Reduction) -> ElementwiseReduction<Reduction>;

template <typename Reduction>
ElementwiseReduction(ReductionWrapper<Reduction>) -> ElementwiseReduction<Reduction>;

}  // namespace detail

/**
 * @cond
 *
 * TODO(eniebler) 2024-06-07: Disabling `make_reduction` for now because it
 * fails tests for reasons I don't yet understand.
 */

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Create an object whose type satisfies the `legate_reduction` concept.
 *
 * Use `make_reduction` to create a Legate reduction object from familiar functional objects like
 * `std::plus`, `std::minus`, `std::multiplies`, and `std::divides`; or from your own custom
 * binary callables.
 *
 * To build a Legate reduction object, you must provide two binary callable objects `apply` and
 * `fold`, both of which take two arguments of the same type and return a value of the same type.
 * In addition, you must specify a value for the identity element of the reduction operation.
 *
 * As a convenience, the `fold` operation defaults to the `apply` operation if not provided. This
 * is correct for some reductions, like addition and multiplication, but not for others like
 * subtraction and division.
 *
 * @li `apply(apply(x, y), z) == apply(x, fold(y, z))`
 * @li `apply(x, identity) == x`
 * @li `fold(x, identity) == fold(identity, x) == x`
 *
 * @pre @li The `apply` and `fold` functions must be stateless.
 *
 * @par Example:
 * @snippet{trimleft} experimental/stl/reduce.cc make-reduction-doxy-snippet
 *
 * @see @li @c reduce
 *      @li @c transform_reduce
 *      @li @c legate_reduction
 * @ingroup stl-utilities
 */
// template <typename ValueType, ValueType Identity, typename Apply, typename Fold = Apply>
// constexpr detail::BasicReduction<Identity, Apply, Fold> make_reduction(
//   [[maybe_unused]] Apply apply, [[maybe_unused]] Fold fold = {})
// {
//   static_assert(legate::type_code_of<ValueType> != legate::Type::Code::NIL,
//                 "The value type of the reduction function must be a valid Legate type");
//   static_assert(std::is_invocable_r_v<ValueType, Apply, ValueType, ValueType>,
//                 "The apply function must be callable with two arguments of type ValueType "
//                 "and must return a value of type ValueType");
//   static_assert(std::is_invocable_r_v<ValueType, Fold, ValueType, ValueType>,
//                 "The fold function must be callable with two arguments of type ValueType "
//                 "and must return a value of type ValueType");
//   static_assert(std::is_empty_v<Apply>, "The apply function must be stateless");
//   static_assert(std::is_empty_v<Fold>, "The fold function must be stateless");
//   return {};
// }

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @overload
 *
 * This overload of `make_reduction` deduces the value type from the `Identity` argument.
 *
 * @par Example:
 * @code{.cpp}
 * The following two are equivalent:
 * auto sum1 = stl::make_reduction<int, 0>(std::plus<>{});
 * auto sum2 = stl::make_reduction<0>(std::plus<>{});
 * @endcode
 */
// template <auto Identity, typename Apply, typename Fold = Apply>
// detail::BasicReduction<Identity, Apply, Fold> make_reduction(Apply, Fold = {})
// {
//   using value_type = std::remove_cv_t<decltype(Identity)>;
//   return stl::make_reduction<value_type, Identity>(Apply{}, Fold{});
// }

/** @endcond */

/** @cond */
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ValueType = void, typename Reduction>
  requires(legate_reduction<Reduction>)  //
[[nodiscard]] auto as_reduction(Reduction red)
{
  using RHS = typename Reduction::RHS;
  if constexpr (callable<Reduction, RHS&, RHS>) {
    return red;
  } else {
    return detail::ReductionWrapper{std::move(red)};
  }
}

template <typename ValueType = void, typename T>
[[nodiscard]] auto as_reduction(std::plus<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of_v<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::ReductionWrapper{legate::SumReduction<Type>()};
}

template <typename ValueType = void, typename T>
[[nodiscard]] auto as_reduction(std::minus<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of_v<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::ReductionWrapper{legate::DiffReduction<Type>()};
}

template <typename ValueType = void, typename T>
[[nodiscard]] auto as_reduction(std::multiplies<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of_v<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::ReductionWrapper{legate::ProdReduction<Type>()};
}

template <typename ValueType = void, typename T>
[[nodiscard]] auto as_reduction(std::divides<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of_v<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::ReductionWrapper{legate::DivReduction<Type>()};
}

// TODO(ericniebler): min and max reductions

template <typename ValueType = void, typename T>
[[nodiscard]] auto as_reduction(std::logical_or<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of_v<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::ReductionWrapper{legate::OrReduction<Type>()};
}

template <typename ValueType = void, typename T>
[[nodiscard]] auto as_reduction(std::logical_and<T>)
{
  using Type = std::conditional_t<std::is_void_v<T>, ValueType, T>;
  static_assert(legate::type_code_of_v<Type> != legate::Type::Code::NIL,
                "The value type of the reduction function must be a valid Legate type");
  return detail::ReductionWrapper{legate::AndReduction<Type>()};
}

// TODO(ericniebler): logical xor

template <typename ValueType = void, typename Function>
[[nodiscard]] auto as_reduction(const detail::Elementwise<Function>& fn)
{
  return detail::ElementwiseReduction{stl::as_reduction<ValueType>(fn.function())};
}

template <typename Fun, typename ValueType = void>
using as_reduction_t = decltype(stl::as_reduction<ValueType>(std::declval<Fun>()));
/** @endcond */

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Reduces the elements of the input range using the given reduction
 * operation operation.
 *
 * @param input The input range to reduce.
 * @param init The initial value of the reduction.
 * @param op The reduction operation to apply to the elements of the input
 *          range. `op` can be a type that satisfies the @c legate_reduction
 *          concept or one of the standard functional objects `std::plus`,
 *          `std::minus`, `std::multiplies`, `std::divides`, etc.; or an
 *          elementwise operation created by passing any of the above to
 *          @c stl::elementwise.
 *
 * @pre @li `InputRange` must satisfy the @c logical_store_like concept.
 *      @li `Init` must satisfy the @c logical_store_like concept.
 *      @li The value type of the input range must be the same as the value
 *          type of the initial value.
 *      @li The dimension of the input range must be one greater than the
 *          dimension of the initial value.
 *
 * @return An instance of @c logical_store with the same value type and shape as
 * `init`.
 *
 * @par Examples:
 * @snippet{trimleft} experimental/stl/reduce.cc stl-reduce-1d
 * @snippet{trimleft} experimental/stl/reduce.cc stl-reduce-2d
 *
 * @see @li @c legate_reduction
 *      @li @ref reduction "Legate's built-in reduction operations"
 * @ingroup stl-algorithms
 */
template <typename InputRange, typename Init, typename ReductionOperation>  //
  requires(logical_store_like<InputRange> && logical_store_like<Init> &&
           legate_reduction<as_reduction_t<ReductionOperation, element_type_of_t<Init>>>)  //
[[nodiscard]] auto reduce(InputRange&& input, Init&& init, ReductionOperation op)
  -> logical_store<element_type_of_t<InputRange>, dim_of_v<Init>>
{
  detail::check_function_type<as_reduction_t<ReductionOperation, element_type_of_t<Init>>>();
  static_assert(std::is_same_v<value_type_of_t<InputRange>, value_type_of_t<Init>>);
  static_assert(dim_of_v<InputRange> == dim_of_v<Init> + 1);
  static_assert(std::is_empty_v<ReductionOperation>,
                "Only stateless reduction operations are currently supported");

  // promote the initial value to the same shape as the input so they can
  // be aligned
  using Input       = std::decay_t<InputRange>;
  using InputPolicy = typename Input::policy;

  LogicalStore out =
    InputPolicy::aligned_promote(get_logical_store(input), get_logical_store(init));
  LEGATE_ASSERT(static_cast<std::size_t>(out.dim()) == static_cast<std::size_t>(init.dim() + 1));

  using OutputRange = slice_view<value_type_of_t<Input>, dim_of_v<Input>, InputPolicy>;
  OutputRange output{std::move(out)};  // NOLINT(misc-const-correctness)

  stl::launch_task(
    stl::inputs(std::forward<InputRange>(input)),
    stl::reduction(std::move(output), stl::as_reduction<element_type_of_t<Init>>(std::move(op))),
    stl::constraints(stl::align(stl::reduction, stl::inputs[0])));

  return as_typed<element_type_of_t<Init>, dim_of_v<Init>>(get_logical_store(init));
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
