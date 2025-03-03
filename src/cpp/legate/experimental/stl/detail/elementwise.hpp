/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/config.hpp>
#include <legate/experimental/stl/detail/stlfwd.hpp>
#include <legate/experimental/stl/detail/store.hpp>

#include <cstdint>
#include <functional>
#include <utility>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {
namespace detail {

template <typename Function, typename... InputSpans>
class ElementwiseAccessor {
 public:
  using value_type       = call_result_t<Function, typename InputSpans::reference...>;
  using element_type     = value_type;
  using data_handle_type = std::size_t;
  using reference        = value_type;
  using offset_policy    = ElementwiseAccessor;

  ElementwiseAccessor() noexcept = default;

  LEGATE_HOST_DEVICE explicit ElementwiseAccessor(Function fun, InputSpans... spans) noexcept
    : fun_{std::move(fun)}, spans_{std::move(spans)...}
  {
  }

  LEGATE_HOST_DEVICE [[nodiscard]] reference access(data_handle_type handle,
                                                    std::size_t i) const noexcept
  {
    auto offset = this->offset(handle, i);
    return std::apply(
      [offset, this](auto&&... span) {  //
        return fun_(span.accessor().access(span.data_handle(), span.mapping()(offset))...);
      },
      spans_);
  }

  LEGATE_HOST_DEVICE [[nodiscard]] typename offset_policy::data_handle_type offset(
    data_handle_type handle, std::size_t i) const noexcept
  {
    return handle + i;
  }

  // private:
  Function fun_{};
  std::tuple<InputSpans...> spans_{};
};

template <typename Function, typename... InputSpans>
using elementwise_span =
  std::mdspan<call_result_t<Function, typename InputSpans::reference...>,
              std::dextents<coord_t, meta::front<InputSpans...>::extents_type::rank()>,
              std::layout_right,
              ElementwiseAccessor<Function, InputSpans...>>;

// a binary function that folds its two arguments together using
// the given binary function, and stores the result in the first
template <typename Function>
class Elementwise : private Function {
 public:
  Elementwise() = default;
  explicit Elementwise(Function fn) : Function{std::move(fn)} {}

  [[nodiscard]] const Function& function() const noexcept { return *this; }

  template <typename InputSpan, typename... InputSpans>
  LEGATE_HOST_DEVICE [[nodiscard]] auto operator()(InputSpan&& head, InputSpans&&... tail) const
    -> elementwise_span<Function, as_mdspan_t<InputSpan>, as_mdspan_t<InputSpans>...>
  {
    // TODO(wonchanl): Put back these assertions once we figure out the compile error
    // static_assert((as_mdspan_t<InputSpan>::extents_type::rank() ==
    //                  as_mdspan_t<InputSpans>::extents_type::rank() &&
    //                ...));
    // LEGATE_ASSERT((stl::as_mdspan(head).extents() == stl::as_mdspan(tail).extents() && ...));

    using Mapping = std::layout_right::mapping<
      std::dextents<legate::coord_t, as_mdspan_t<InputSpan>::extents_type::rank()>>;
    using Accessor = stl::detail::
      ElementwiseAccessor<Function, as_mdspan_t<InputSpan>, as_mdspan_t<InputSpans>...>;
    using ElementwiseSpan =
      elementwise_span<Function, as_mdspan_t<InputSpan>, as_mdspan_t<InputSpans>...>;

    // These can *sometimes* be moved
    // NOLINTBEGIN(misc-const-correctness)
    Mapping mapping{head.extents()};
    Accessor accessor{function(),
                      stl::as_mdspan(std::forward<InputSpan>(head)),
                      stl::as_mdspan(std::forward<InputSpans>(tail))...};
    // NOLINTEND(misc-const-correctness)
    return ElementwiseSpan{0, std::move(mapping), std::move(accessor)};
  }
};

}  // namespace detail

/**
 * @brief A functional adaptor that, given a callable object `fn`, returns
 * another callable object `g` that applies `fn` element-wise to its arguments.
 *
 * The arguments to `g` must be `mdspan` objects or models of the
 * `logical_store_like` concept. The shapes of the input arguments must
 * all match. The element-wise application of `f` is performed lazily; @em i.e.,
 * the result is not computed until the elements of the result are accessed.
 *
 * @param fn The callable object to apply element-wise.
 *
 * @return A callable object @f$\mathtt{g}@f$ such that, given multi-dimensional
 * arguments @f$A^1,A^2\cdots,A^n@f$, the expression @f$\mathtt{g(}A^1,A^2\cdots,A^n\mathtt{)}@f$
 * returns a multi-dimensional view @f$\mathtt{V}@f$ where @f$\mathtt{V}_{i,j,\ldots}@f$ is
 * the result of calling
 * @f$\mathtt{fn(}{A^1}_{i,j,\ldots}, {A^2}_{i,j,\ldots}, \cdots, {A^n}_{i,j,\ldots}\mathtt{)}@f$.
 *
 * @note The Legate.STL algorithms recognize the return type of
 * @f$\mathtt{elementwise(fn)(}A^1,A^2\cdots,A^n\mathtt{)}@f$ such that assigning its result
 * to an `mdspan` object will perform an element-wise assignment. The element-wise assignment
 * is done with `thrust::copy` and will be accelerated if CUDA support is enabled.
 *
 * @par Example:
 * @snippet{trimleft} experimental/stl/elementwise.cc elementwise example
 *
 * @ingroup stl-utilities
 */
template <typename Function>
[[nodiscard]] LEGATE_STL_UNSPECIFIED(detail::Elementwise<std::decay_t<Function>>)
  elementwise(Function&& fn)
{
  return detail::Elementwise<std::decay_t<Function>>{std::forward<Function>(fn)};
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
