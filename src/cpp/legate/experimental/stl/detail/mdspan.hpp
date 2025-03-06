/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/cuda/cuda.h>
#include <legate/experimental/stl/detail/config.hpp>  // includes <version>
#include <legate/utilities/assert.h>
#include <legate/utilities/macros.h>

#if __has_include(<mdspan>)
#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L
#define LEGATE_STL_HAS_STD_MDSPAN
#endif
#endif

#if LEGATE_DEFINED(LEGATE_STL_HAS_STD_MDSPAN)

#include <mdspan>

#else

LEGATE_PRAGMA_PUSH();
LEGATE_PRAGMA_EDG_IGNORE(
  737,
  useless_using_declaration,  // using-declaration ignored -- it refers to the current namespace
  20011,
  20040,   // a __host__ function [...] redeclared with __host__ __device__
  20014);  // calling a __host__ function [...] from a __host__ __device__
           // function is not allowed

#include <legate/experimental/stl/detail/span.hpp>  // this header must come before mdspan.hpp

// Blame Kokkos for these uses of reserved identifiers...
// NOLINTBEGIN
#define _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS 1
#define _MDSPAN_USE_FAKE_ATTRIBUTE_NO_UNIQUE_ADDRESS
#define _MDSPAN_NO_UNIQUE_ADDRESS
// NOLINTEND

#define MDSPAN_IMPL_STANDARD_NAMESPACE std
#define MDSPAN_IMPL_PROPOSED_NAMESPACE mdspan_experimental
#include <mdspan/mdspan.hpp>
// We intentionally define this so that downstream libs do the right thing.
#define __cpp_lib_mdspan 1  // NOLINT

namespace std {
// DANGER: this actually is potentially quite dangerous...
// NOLINTNEXTLINE(google-build-using-namespace, cert-dcl58-cpp)
using namespace mdspan_experimental;
}  // namespace std

LEGATE_PRAGMA_POP();

#endif  // LEGATE_STL_HAS_STD_MDSPAN

// Legate includes:
#include <legate.h>

#include <legate/cuda/cuda.h>
#include <legate/experimental/stl/detail/meta.hpp>
#include <legate/experimental/stl/detail/type_traits.hpp>

// NVIDIA includes:
#include <nv/target>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

// Standard includes:
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
  std::mdspan<ElementType,
              std::dextents<coord_t, Dim>,
              std::layout_right,
              detail::MDSpanAccessor<ElementType, Dim>>;

template <typename Op, std::int32_t Dim, bool Exclusive = false>
using mdspan_reduction_t =  //
  std::mdspan<
    typename Op::RHS,
    std::dextents<coord_t, Dim>,
    std::layout_right,
    detail::MDSpanAccessor<typename Op::RHS, Dim, detail::ReductionAccessor<Op, Exclusive>>>;

namespace detail {

template <typename T>
inline constexpr bool is_mdspan_v = false;

template <typename T>
inline constexpr bool is_mdspan_v<T&> = is_mdspan_v<T>;

template <typename T>
inline constexpr bool is_mdspan_v<T const> = is_mdspan_v<T>;

template <typename Element, typename Extent, typename Layout, typename Accessor>
inline constexpr bool is_mdspan_v<std::mdspan<Element, Extent, Layout, Accessor>> = true;

template <typename MDSpan>
class flat_mdspan_view;

/**
 * @brief An iterator over an `mdspan` that presents a flat view and allows
 * random elementwise access. It is particularly handy for passing to Thrust
 * algorithms to perform elementwise operations in parallel.
 *
 * @ingroup stl-views
 */
template <typename MDSpan>
class flat_mdspan_iterator;

template <typename Element, typename Extent, typename Layout, typename Accessor>
class flat_mdspan_iterator<std::mdspan<Element, Extent, Layout, Accessor>> {
  using mdspan_t         = std::mdspan<Element, Extent, Layout, Accessor>;
  using data_handle_type = typename mdspan_t::data_handle_type;
  friend class flat_mdspan_view<mdspan_t>;

  struct pointer_type {  // NOLINT(readability-identifier-naming)
    std::remove_const_t<Element> elem_;
    LEGATE_HOST_DEVICE [[nodiscard]] Element* operator->() const noexcept { return &elem_; }
  };

 public:
  using value_type        = std::remove_const_t<Element>;
  using reference         = typename mdspan_t::reference;
  using difference_type   = std::ptrdiff_t;
  using iterator_category = std::random_access_iterator_tag;
  using pointer           = std::conditional_t<std::is_lvalue_reference_v<reference>,
                                               std::add_pointer_t<reference>,
                                               pointer_type>;

  LEGATE_HOST_DEVICE [[nodiscard]] reference operator*() const noexcept
  {
    static_assert(noexcept(span_->accessor().access(span_->data_handle(), span_->mapping()(idx_))));
    return span_->accessor().access(span_->data_handle(), span_->mapping()(idx_));
  }

  LEGATE_HOST_DEVICE [[nodiscard]] pointer operator->() const noexcept
  {
    if constexpr (std::is_lvalue_reference_v<reference>) {
      return &operator*();
    } else {
      return pointer_type{operator*()};
    }
  }

  LEGATE_HOST_DEVICE flat_mdspan_iterator& operator++() noexcept
  {
    LEGATE_ASSERT(idx_ < span_->size());
    ++idx_;
    return *this;
  }

  LEGATE_HOST_DEVICE flat_mdspan_iterator& operator++(int) noexcept
  {
    auto copy = *this;
    ++(*this);
    return copy;
  }

  LEGATE_HOST_DEVICE flat_mdspan_iterator& operator--() noexcept
  {
    LEGATE_ASSERT(idx_ > 0);
    --idx_;
    return *this;
  }

  LEGATE_HOST_DEVICE flat_mdspan_iterator& operator--(int) noexcept
  {
    auto copy = *this;
    --(*this);
    return copy;
  }

  LEGATE_HOST_DEVICE flat_mdspan_iterator& operator+=(difference_type n) noexcept
  {
    if (n < 0) {
      LEGATE_ASSERT(idx_ >= static_cast<std::size_t>(-n));
      idx_ -= static_cast<std::size_t>(-n);
    } else {
      LEGATE_ASSERT(idx_ + static_cast<std::size_t>(n) <= span_->size());
      idx_ += static_cast<std::size_t>(n);
    }
    return *this;
  }

  LEGATE_HOST_DEVICE flat_mdspan_iterator& operator-=(difference_type n) noexcept
  {
    return operator+=(-n);
  }

  LEGATE_HOST_DEVICE [[nodiscard]] reference operator[](difference_type n) const noexcept
  {
    return *(*this + n);
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend difference_type operator-(
    const flat_mdspan_iterator& self, const flat_mdspan_iterator& other) noexcept
  {
    LEGATE_ASSERT(self.span_ == other.span_);
    return static_cast<difference_type>(self.idx_) - static_cast<difference_type>(other.idx_);
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend flat_mdspan_iterator operator+(flat_mdspan_iterator self,
                                                                         difference_type n) noexcept
  {
    self += n;
    return self;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend flat_mdspan_iterator operator+(
    difference_type n, flat_mdspan_iterator self) noexcept
  {
    self += n;
    return self;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend flat_mdspan_iterator operator-(flat_mdspan_iterator self,
                                                                         difference_type n) noexcept
  {
    self -= n;
    return self;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator==(const flat_mdspan_iterator& lhs,
                                                          const flat_mdspan_iterator& rhs) noexcept
  {
    LEGATE_ASSERT(lhs.span_ == rhs.span_);
    return lhs.idx_ == rhs.idx_;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator!=(const flat_mdspan_iterator& lhs,
                                                          const flat_mdspan_iterator& rhs) noexcept
  {
    return !(lhs == rhs);
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator<(const flat_mdspan_iterator& lhs,
                                                         const flat_mdspan_iterator& rhs) noexcept
  {
    LEGATE_ASSERT(lhs.span_ == rhs.span_);
    return lhs.idx_ < rhs.idx_;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator>(const flat_mdspan_iterator& lhs,
                                                         const flat_mdspan_iterator& rhs) noexcept
  {
    return rhs < lhs;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator<=(const flat_mdspan_iterator& lhs,
                                                          const flat_mdspan_iterator& rhs) noexcept
  {
    return !(rhs < lhs);
  }

  LEGATE_HOST_DEVICE [[nodiscard]] friend bool operator>=(const flat_mdspan_iterator& lhs,
                                                          const flat_mdspan_iterator& rhs) noexcept
  {
    return !(lhs < rhs);
  }

 private:
  LEGATE_HOST_DEVICE explicit flat_mdspan_iterator(const mdspan_t* span, std::size_t idx) noexcept
    : span_{span}, idx_(idx)
  {
  }

  const mdspan_t* span_{};
  std::size_t idx_{};
};

/**
 * @brief A flattened view of an `mdspan` that allows efficient random
 * elementwise access.
 *
 * @ingroup stl-views
 */
template <typename Element, typename Extent, typename Layout, typename Accessor>
class flat_mdspan_view<std::mdspan<Element, Extent, Layout, Accessor>> {
 public:
  using mdspan_t       = std::mdspan<Element, Extent, Layout, Accessor>;
  using iterator       = flat_mdspan_iterator<mdspan_t>;
  using const_iterator = iterator;

  LEGATE_HOST_DEVICE explicit flat_mdspan_view(mdspan_t span) noexcept : span_{std::move(span)} {}

  LEGATE_HOST_DEVICE [[nodiscard]] iterator begin() const noexcept { return iterator{&span_, 0}; }

  LEGATE_HOST_DEVICE [[nodiscard]] iterator end() const noexcept
  {
    return iterator{&span_, span_.size()};
  }

 private:
  mdspan_t span_;
};

/**
 * @brief Create a flattened view of an `mdspan` that allows efficient random
 * elementwise access.
 *
 * @ingroup stl-views
 */
template <typename Element, typename Extent, typename Layout, typename Accessor>
LEGATE_HOST_DEVICE [[nodiscard]] flat_mdspan_view<std::mdspan<Element, Extent, Layout, Accessor>>
flatten(std::mdspan<Element, Extent, Layout, Accessor> span) noexcept
{
  return flat_mdspan_view<std::mdspan<Element, Extent, Layout, Accessor>>{std::move(span)};
}

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
          typename Layout,
          typename LeftAccessor,
          typename RightAccessor>
LEGATE_HOST_DEVICE void assign(std::mdspan<LeftElement, Extent, Layout, LeftAccessor>&& lhs,
                               std::mdspan<RightElement, Extent, Layout, RightAccessor>&& rhs)
{
  static_assert(
    std::is_assignable_v<typename LeftAccessor::reference, typename RightAccessor::reference>);
  LEGATE_ASSERT(lhs.extents() == rhs.extents());

  const auto lhs_view = detail::flatten(std::move(lhs));
  const auto rhs_view = detail::flatten(std::move(rhs));

  LEGATE_PRAGMA_PUSH();
  LEGATE_PRAGMA_CLANG_IGNORE("-Wgnu-zero-variadic-macro-arguments");
  NV_IF_TARGET(NV_IS_HOST,
               (thrust::copy(thrust::host, rhs_view.begin(), rhs_view.end(), lhs_view.begin());),
               (thrust::copy(thrust::device, rhs_view.begin(), rhs_view.end(), lhs_view.begin());))
  LEGATE_PRAGMA_POP();
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
