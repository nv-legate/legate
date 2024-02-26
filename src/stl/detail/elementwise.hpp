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

#include "config.hpp"
#include "stlfwd.hpp"
#include "store.hpp"

#include <cstdint>
#include <functional>
#include <utility>

namespace legate::stl {
namespace detail {
template <typename Function, typename... InputSpans>
class elementwise_accessor {
 public:
  using value_type       = call_result_t<Function, typename InputSpans::reference...>;
  using element_type     = value_type;
  using data_handle_type = std::size_t;
  using reference        = value_type;
  using offset_policy    = elementwise_accessor;

  // LEGATE_STL_ATTRIBUTE((host, device)) //
  elementwise_accessor() noexcept = default;

  LEGATE_STL_ATTRIBUTE((host, device))
  explicit elementwise_accessor(Function fun, InputSpans... spans) noexcept
    : fun_(std::move(fun)), spans_(std::move(spans)...)
  {
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  reference access(data_handle_type handle, std::size_t i) const noexcept
  {
    auto offset = this->offset(handle, i);
    return std::apply(
      [offset, this](auto&&... span) {  //
        return fun_(span.accessor().access(span.data_handle(), span.mapping()(offset))...);
      },
      spans_);
  }

  LEGATE_STL_ATTRIBUTE((host, device))
  typename offset_policy::data_handle_type offset(data_handle_type handle,
                                                  std::size_t i) const noexcept
  {
    return handle + i;
  }

  // private:
  Function fun_;
  std::tuple<InputSpans...> spans_;
};

template <typename Function, typename... InputSpans>
using elementwise_span =
  std::mdspan<call_result_t<Function, typename InputSpans::reference...>,
              std::dextents<coord_t, meta::front<InputSpans...>::extents_type::rank()>,
              std::layout_right,
              elementwise_accessor<Function, InputSpans...>>;

// a binary function that folds its two arguments together using
// the given binary function, and stores the result in the first
template <typename Function>
class elementwise {
 public:
  LEGATE_STL_ATTRIBUTE((no_unique_address)) Function fn;

  template <typename InputSpan, typename... InputSpans>
  LEGATE_STL_ATTRIBUTE((host, device))  //
  auto operator()(InputSpan&& head, InputSpans&&... tail) const
    -> elementwise_span<Function, as_mdspan_t<InputSpan>, as_mdspan_t<InputSpans>...>
  {
    // TODO(wonchanl): Put back these assertions once we figure out the compile error
    // static_assert((as_mdspan_t<InputSpan>::extents_type::rank() ==
    //                  as_mdspan_t<InputSpans>::extents_type::rank() &&
    //                ...));
    // LegateAssert((stl::as_mdspan(head).extents() == stl::as_mdspan(tail).extents() && ...));

    using Mapping = std::layout_right::mapping<
      std::dextents<legate::coord_t, as_mdspan_t<InputSpan>::extents_type::rank()>>;
    using Accessor = stl::detail::
      elementwise_accessor<Function, as_mdspan_t<InputSpan>, as_mdspan_t<InputSpans>...>;
    using ElementwiseSpan =
      elementwise_span<Function, as_mdspan_t<InputSpan>, as_mdspan_t<InputSpans>...>;

    Mapping mapping{head.extents()};
    Accessor accessor{fn, stl::as_mdspan(head), stl::as_mdspan(tail)...};
    return ElementwiseSpan{0, mapping, accessor};
  }
};

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Function>
detail::elementwise<Function> elementwise(Function fn)
{
  return {std::move(fn)};
}

}  // namespace legate::stl
