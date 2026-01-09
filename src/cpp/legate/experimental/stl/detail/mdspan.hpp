/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/experimental/stl/detail/config.hpp>  // includes <version>
#include <legate/utilities/assert.h>
#include <legate/utilities/mdspan.h>

#include <cuda/std/mdspan>

// Legate includes:
#include <legate.h>

#include <legate/experimental/stl/detail/meta.hpp>
#include <legate/experimental/stl/detail/type_traits.hpp>

// Standard includes:
#include <algorithm>
#include <cstdint>
#include <type_traits>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {
namespace detail {

template <typename Function, typename... InputSpans>
class ElementwiseAccessor;

template <Legion::PrivilegeMode Privilege, typename ElementType, std::int32_t Dim>
using store_accessor_t =  //
  Legion::FieldAccessor<Privilege,
                        ElementType,
                        Dim,
                        Legion::coord_t,
                        Legion::AffineAccessor<ElementType, Dim>>;

class DefaultAccessor {
 public:
  template <typename ElementType, std::int32_t Dim>
  using type =  //
    meta::if_c<(Dim == 0),
               store_accessor_t<LEGION_READ_ONLY, const ElementType, 1>,
               meta::if_c<std::is_const_v<ElementType>,
                          store_accessor_t<LEGION_READ_ONLY, ElementType, Dim>,
                          store_accessor_t<LEGION_READ_WRITE, ElementType, Dim>>>;

  // If an exception is thrown here, then we are well and truly screwed anyways, so may as well
  // have the compiler abort
  // NOLINTBEGIN(bugprone-exception-escape)
  template <typename ElementType, std::int32_t Dim>
  LEGATE_HOST_DEVICE [[nodiscard]] static type<ElementType, Dim> get(
    const PhysicalStore& store) noexcept
  {
    if constexpr (Dim == 0) {
      // 0-dimensional legate stores are backed by read-only futures
      LEGATE_ASSERT(store.is_future());
      return store.read_accessor<const ElementType, 1>();
    } else if constexpr (std::is_const_v<ElementType>) {
      return store.read_accessor<ElementType, Dim>();
    } else {
      return store.read_write_accessor<ElementType, Dim>();
    }
  }

  // NOLINTEND(bugprone-exception-escape)
};

template <typename Op, bool Exclusive = false>
class ReductionAccessor {
 public:
  template <typename ElementType, std::int32_t Dim>
  using type =  //
    meta::if_c<(Dim == 0),
               store_accessor_t<LEGION_READ_ONLY, const ElementType, 1>,
               Legion::ReductionAccessor<Op,
                                         Exclusive,
                                         Dim,
                                         coord_t,
                                         Realm::AffineAccessor<typename Op::RHS, Dim, coord_t>>>;

  // If an exception is thrown here, then we are well and truly screwed anyways, so may as well
  // have the compiler abort
  // NOLINTBEGIN(bugprone-exception-escape)
  template <typename ElementType, std::int32_t Dim>
  LEGATE_HOST_DEVICE [[nodiscard]] static type<ElementType, Dim> get(
    const PhysicalStore& store) noexcept
  {
    if constexpr (Dim == 0) {
      // 0-dimensional legate stores are backed by read-only futures
      LEGATE_ASSERT(store.is_future());
      return store.read_accessor<const ElementType, 1>();
    } else {
      return store.reduce_accessor<Op, Exclusive, Dim>();
    }
  }

  // NOLINTEND(bugprone-exception-escape)
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// mdspan_accessor:
//    A custom accessor policy for use with std::mdspan for accessing a Legate store.
template <typename ElementType, std::int32_t ActualDim, typename Accessor = DefaultAccessor>
class MDSpanAccessor {
 public:
  static constexpr auto DIM = std::max(ActualDim, std::int32_t{1});
  using value_type          = std::remove_const_t<ElementType>;
  using element_type        = ElementType;
  using data_handle_type    = std::size_t;
  using accessor_type       = typename Accessor::template type<ElementType, ActualDim>;
  using reference           = decltype(std::declval<const accessor_type&>()[Point<DIM>::ONES()]);
  using offset_policy       = MDSpanAccessor;

  template <typename, std::int32_t, typename>
  friend class mdspan_accessor;

  // NOLINTNEXTLINE(modernize-use-equals-default):  to work around an nvcc-11 bug
  LEGATE_HOST_DEVICE MDSpanAccessor() noexcept  // = default;
  {
  }

  LEGATE_HOST_DEVICE explicit MDSpanAccessor(PhysicalStore store, const Rect<DIM>& shape) noexcept
    : store_{std::move(store)},
      shape_{shape.hi - shape.lo + Point<DIM>::ONES()},
      origin_{shape.lo},
      accessor_{Accessor::template get<ElementType, ActualDim>(store_)}
  {
  }

  LEGATE_HOST_DEVICE explicit MDSpanAccessor(const PhysicalStore& store) noexcept
    : MDSpanAccessor{store, store.shape<DIM>()}
  {
  }

  // Need this specifically for GCC only, since clang does not understand maybe-uninitialized
  // (but it also doesn't have a famously broken "maybe uninitialized" checker...).
  //
  // This ignore is needed to silence the following spurious warnings, because I guess the
  // Kokkos guys don't default-initialize their compressed pairs?
  //
  // legate/src/core/experimental/stl/detail/mdspan.hpp:171:3: error:
  // '<unnamed>.std::detail::__compressed_pair<std::layout_right::mapping<std::extents<long
  // long int, 18446744073709551615> >, legate::experimental::stl::detail::mdspan_accessor<long int,
  // 1, legate::experimental::stl::detail::default_accessor>,
  // void>::__t2_val.legate::experimental::stl::detail::mdspan_accessor<long int, 1,
  // legate::experimental::stl::detail::default_accessor>::shape_' may be used uninitialized
  // [-Werror=maybe-uninitialized]
  // 171 |   mdspan_accessor(mdspan_accessor&& other) noexcept = default;
  //     |   ^~~~~~~~~~~~~~~
  //
  // legate/arch-ci-linux-gcc-py-pkgs-release/cmake_build/_deps/mdspan-src/include/experimental/__p0009_bits/mdspan.hpp:198:36:
  // note: '<anonymous>' declared here
  //   198 |     : __members(other.__ptr_ref(), __map_acc_pair_t(other.__mapping_ref(),
  //   other.__accessor_ref()))
  //       |                                    ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  LEGATE_PRAGMA_PUSH();
  LEGATE_PRAGMA_GCC_IGNORE("-Wmaybe-uninitialized");
  LEGATE_HOST_DEVICE MDSpanAccessor(MDSpanAccessor&& other) noexcept = default;
  LEGATE_HOST_DEVICE MDSpanAccessor(const MDSpanAccessor& other)     = default;
  LEGATE_PRAGMA_POP();

  LEGATE_HOST_DEVICE MDSpanAccessor& operator=(MDSpanAccessor&& other) noexcept
  {
    *this = other;
    return *this;
  }

  LEGATE_HOST_DEVICE MDSpanAccessor& operator=(const MDSpanAccessor& other) noexcept
  {
    if (this == &other) {
      return *this;
    }
    store_    = other.store_;
    shape_    = other.shape_;
    origin_   = other.origin_;
    accessor_ = other.accessor_;
    return *this;
  }

  // NOLINTBEGIN(google-explicit-constructor)
  template <typename OtherElementType>                                          //
    requires(std::is_convertible_v<OtherElementType (*)[], ElementType (*)[]>)  //
  LEGATE_HOST_DEVICE MDSpanAccessor(
    const MDSpanAccessor<OtherElementType, DIM, Accessor>& other) noexcept
    : store_{other.store_},
      shape_{other.shape_},
      origin_{other.origin_},
      accessor_{Accessor::template get<ElementType, ActualDim>(store_)}
  {
  }

  // NOLINTEND(google-explicit-constructor)

  LEGATE_HOST_DEVICE [[nodiscard]] reference access(data_handle_type handle,
                                                    std::size_t i) const noexcept
  {
    Point<DIM> p;
    auto offset = handle + i;

    for (auto dim = DIM - 1; dim >= 0; --dim) {
      p[dim] = offset % shape_[dim];
      offset /= shape_[dim];
    }
    return accessor_[p + origin_];
  }

  LEGATE_HOST_DEVICE [[nodiscard]] typename offset_policy::data_handle_type offset(
    data_handle_type handle, std::size_t i) const noexcept
  {
    return handle + i;
  }

 private:
  PhysicalStore store_{nullptr};
  Point<DIM> shape_{};
  Point<DIM> origin_{};
  accessor_type accessor_{};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Store>
struct ValueTypeOf : meta::if_c<(type_code_of_v<Store> == Type::Code::NIL),
                                meta::empty,
                                legate::detail::type_identity<Store>> {};

}  // namespace detail

/**
 * @brief An alias for `std::mdspan` with a custom accessor that allows
 *       elementwise access to a `legate::PhysicalStore`.
 *
 * @tparam ElementType The element type of the `mdspan`.
 * @tparam Dim The dimensionality of the `mdspan`.
 *
 * @ingroup stl-views
 */
template <typename ElementType, std::int32_t Dim>
using mdspan_t =  //
  ::cuda::std::mdspan<ElementType,
                      ::cuda::std::dextents<coord_t, Dim>,
                      ::cuda::std::layout_right,
                      detail::MDSpanAccessor<ElementType, Dim>>;

template <typename Op, std::int32_t Dim, bool Exclusive = false>
using mdspan_reduction_t =  //
  ::cuda::std::mdspan<
    typename Op::RHS,
    ::cuda::std::dextents<coord_t, Dim>,
    ::cuda::std::layout_right,
    detail::MDSpanAccessor<typename Op::RHS, Dim, detail::ReductionAccessor<Op, Exclusive>>>;

namespace detail {

template <typename T>
inline constexpr bool is_mdspan_v = false;

template <typename T>
inline constexpr bool is_mdspan_v<T&> = is_mdspan_v<T>;

template <typename T>
inline constexpr bool is_mdspan_v<T const> = is_mdspan_v<T>;

template <typename Element, typename Extent, typename Layout, typename Accessor>
inline constexpr bool is_mdspan_v<::cuda::std::mdspan<Element, Extent, Layout, Accessor>> = true;

}  // namespace detail

template <typename LHS, typename RHS>
LEGATE_HOST_DEVICE void assign(LHS&& lhs, RHS&& rhs)
{
  static_assert(!detail::is_mdspan_v<LHS> && !detail::is_mdspan_v<RHS>);
  static_assert(std::is_assignable_v<LHS, RHS>);
  static_cast<LHS&&>(lhs) = static_cast<RHS&&>(rhs);
}

template <typename LeftElement,
          typename RightElement,
          typename Extent,
          typename LeftLayout,
          typename RightLayout,
          typename LeftAccessor,
          typename RightAccessor>
LEGATE_HOST_DEVICE void assign(
  ::cuda::std::mdspan<LeftElement, Extent, LeftLayout, LeftAccessor>&& lhs,
  ::cuda::std::mdspan<RightElement, Extent, RightLayout, RightAccessor>&& rhs)
{
  static_assert(
    std::is_assignable_v<typename LeftAccessor::reference, typename RightAccessor::reference>);
  LEGATE_ASSERT(lhs.extents() == rhs.extents());

  for_each_in_extent(
    rhs.extents(), [=](auto... indices) LEGATE_HOST_DEVICE { lhs(indices...) = rhs(indices...); });
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
