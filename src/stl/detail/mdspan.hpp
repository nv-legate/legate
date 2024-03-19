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
#include "core/utilities/defined.h"

#include "config.hpp"  // includes <version>

#if __has_include(<mdspan>)
#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L
#define LEGATE_STL_HAS_STD_MDSPAN
#endif
#endif

#if LegateDefined(LEGATE_STL_HAS_STD_MDSPAN)

#include <mdspan>

#else

LEGATE_STL_PRAGMA_PUSH()
LEGATE_STL_PRAGMA_EDG_IGNORE(
  737,
  useless_using_declaration,  // using-declaration ignored -- it refers to the current namespace
  20011,
  20040,  // a __host__ function [...] redeclared with __host__ __device__
  20014)  // calling a __host__ function [...] from a __host__ __device__
          // function is not allowed

#include "span.hpp"  // this header must come before mdspan.hpp

// Blame Kokkos for these uses of reserved identifiers...
// NOLINTBEGIN(bugprone-reserved-identifier)
#define _MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS 1
#define _MDSPAN_USE_FAKE_ATTRIBUTE_NO_UNIQUE_ADDRESS
#define _MDSPAN_NO_UNIQUE_ADDRESS
// NOLINTEND(bugprone-reserved-identifier)

#define MDSPAN_IMPL_STANDARD_NAMESPACE std
#define MDSPAN_IMPL_PROPOSED_NAMESPACE mdspan_experimental
#include <mdspan/mdspan.hpp>
// We intentionally define this so that downstream libs do the right thing.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define __cpp_lib_mdspan 1

namespace std {
// DANGER: this actually is potentially quite dangerous...
// NOLINTNEXTLINE(google-build-using-namespace, cert-dcl58-cpp)
using namespace mdspan_experimental;
}  // namespace std

LEGATE_STL_PRAGMA_POP()

#endif  // LEGATE_STL_HAS_STD_MDSPAN

#include "legate.h"
#include "meta.hpp"
#include "type_traits.hpp"

#include <cstdint>

// Include this last:
#include "prefix.hpp"

namespace legate::stl {
namespace detail {

template <typename Function, typename... InputSpans>
class elementwise_accessor;

template <Legion::PrivilegeMode Privilege, typename ElementType, std::int32_t Dim>
using store_accessor_t =  //
  Legion::FieldAccessor<Privilege,
                        ElementType,
                        Dim,
                        Legion::coord_t,
                        Legion::AffineAccessor<ElementType, Dim>>;

class default_accessor {
 public:
  template <typename ElementType, std::int32_t Dim>
  using type =  //
    meta::if_c<(Dim == 0),
               store_accessor_t<LEGION_READ_ONLY, const ElementType, 1>,
               meta::if_c<std::is_const_v<ElementType>,
                          store_accessor_t<LEGION_READ_ONLY, ElementType, Dim>,
                          store_accessor_t<LEGION_READ_WRITE, ElementType, Dim>>>;

  template <typename ElementType, std::int32_t Dim>
  LEGATE_HOST_DEVICE [[nodiscard]] static type<ElementType, Dim> get(
    const PhysicalStore& store) noexcept
  {
    if constexpr (Dim == 0) {
      // 0-dimensional legate stores are backed by read-only futures
      LegateAssert(store.is_future());
      return store.read_accessor<const ElementType, 1>();
    } else if constexpr (std::is_const_v<ElementType>) {
      return store.read_accessor<ElementType, Dim>();
    } else {
      return store.read_write_accessor<ElementType, Dim>();
    }
  }
};

template <typename Op, bool Exclusive = false>
class reduction_accessor {
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

  template <typename ElementType, std::int32_t Dim>
  LEGATE_HOST_DEVICE [[nodiscard]] static type<ElementType, Dim> get(
    const PhysicalStore& store) noexcept
  {
    if constexpr (Dim == 0) {
      // 0-dimensional legate stores are backed by read-only futures
      LegateAssert(store.is_future());
      return store.read_accessor<const ElementType, 1>();
    } else {
      return store.reduce_accessor<Op, Exclusive, Dim>();
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// mdspan_accessor:
//    A custom accessor policy for use with std::mdspan for accessing a Legate store.
template <typename ElementType, std::int32_t ActualDim, typename Accessor = default_accessor>
class mdspan_accessor {
 public:
  static constexpr auto Dim = std::max(ActualDim, std::int32_t{1});
  using value_type          = std::remove_const_t<ElementType>;
  using element_type        = const_if_t<ActualDim == 0, ElementType>;
  using data_handle_type    = std::size_t;
  using accessor_type       = typename Accessor::template type<ElementType, ActualDim>;
  using reference           = decltype(std::declval<const accessor_type&>()[Point<Dim>::ONES()]);
  using offset_policy       = mdspan_accessor;

  template <typename, std::int32_t, typename>
  friend class mdspan_accessor;

  // NOLINTNEXTLINE(modernize-use-equals-default):  to work around an nvcc-11 bug
  LEGATE_HOST_DEVICE mdspan_accessor() noexcept  // = default;
  {
  }

  LEGATE_HOST_DEVICE explicit mdspan_accessor(PhysicalStore store, const Rect<Dim>& shape) noexcept
    : store_{std::move(store)},
      shape_{shape.hi - shape.lo + Point<Dim>::ONES()},
      origin_{shape.lo},
      accessor_{Accessor::template get<ElementType, ActualDim>(store_)}
  {
  }

  LEGATE_HOST_DEVICE explicit mdspan_accessor(const PhysicalStore& store) noexcept
    : mdspan_accessor{store, store.shape<Dim>()}
  {
  }

  // Need this specifically for GCC only, since clang does not understand maybe-uninitialized
  // (but it also doesn't have a famously broken "maybe uninitialized" checker...).
  //
  // This ignore is needed to silence the following spurious warnings, because I guess the
  // Kokkos guys don't default-initialize their compressed pairs?
  //
  // legate.core.internal/src/stl/detail/mdspan.hpp:171:3: error:
  // '<unnamed>.std::detail::__compressed_pair<std::layout_right::mapping<std::extents<long
  // long int, 18446744073709551615> >, legate::stl::detail::mdspan_accessor<long int, 1,
  // legate::stl::detail::default_accessor>,
  // void>::__t2_val.legate::stl::detail::mdspan_accessor<long int, 1,
  // legate::stl::detail::default_accessor>::shape_' may be used uninitialized
  // [-Werror=maybe-uninitialized]
  // 171 |   mdspan_accessor(mdspan_accessor&& other) noexcept = default;
  //     |   ^~~~~~~~~~~~~~~
  //
  // legate.core.internal/arch-ci-linux-gcc-py-pkgs-release/cmake_build/_deps/mdspan-src/include/experimental/__p0009_bits/mdspan.hpp:198:36:
  // note: '<anonymous>' declared here
  //   198 |     : __members(other.__ptr_ref(), __map_acc_pair_t(other.__mapping_ref(),
  //   other.__accessor_ref()))
  //       |                                    ^~~~~~~~~~~~~~~~~~~~~~~~~~~
#if LegateDefined(LEGATE_STL_GCC())
  LEGATE_STL_PRAGMA_PUSH()
  LEGATE_STL_PRAGMA_GNU_IGNORE("-Wmaybe-uninitialized")
#endif
  LEGATE_HOST_DEVICE mdspan_accessor(mdspan_accessor&& other) noexcept = default;
  LEGATE_HOST_DEVICE mdspan_accessor(const mdspan_accessor& other)     = default;
#if LegateDefined(LEGATE_STL_GCC())
  LEGATE_STL_PRAGMA_POP()
#endif

  LEGATE_HOST_DEVICE mdspan_accessor& operator=(mdspan_accessor&& other) noexcept
  {
    *this = other;
    return *this;
  }

  LEGATE_HOST_DEVICE mdspan_accessor& operator=(const mdspan_accessor& other) noexcept
  {
    if (this == &other) {
      return *this;
    }
    LegateAssert(elementwise_ == nullptr);
    if (other.elementwise_) {
      other.elementwise_(*this, other.elementwise_obj_);
    } else {
      store_    = other.store_;
      shape_    = other.shape_;
      origin_   = other.origin_;
      accessor_ = other.accessor_;
    }
    return *this;
  }

  // NOLINTBEGIN(google-explicit-constructor)
  template <typename OtherElementType>                                          //
    requires(std::is_convertible_v<OtherElementType (*)[], ElementType (*)[]>)  //
  LEGATE_HOST_DEVICE mdspan_accessor(
    const mdspan_accessor<OtherElementType, Dim, Accessor>& other) noexcept
    : store_{other.store_},
      shape_{other.shape_},
      origin_{other.origin_},
      accessor_{Accessor::template get<ElementType, ActualDim>(store_)}
  {
  }

  template <typename Function, typename... InputSpans>
  LEGATE_HOST_DEVICE mdspan_accessor(const elementwise_accessor<Function, InputSpans...>& other)
    : elementwise_{&mdspan_accessor::elementwise<Function, InputSpans...>}, elementwise_obj_{&other}
  {
  }
  // NOLINTEND(google-explicit-constructor)

  // Apply a function element-wise
  template <typename Function, typename... InputSpans>
  LEGATE_HOST_DEVICE mdspan_accessor& operator=(
    const elementwise_accessor<Function, InputSpans...>& other)
  {
    using result_type = call_result_t<Function, typename InputSpans::reference...>;
    static_assert(
      std::is_assignable_v<reference, result_type>,
      "The elementwise function must return a type that is assignable to the reference type of "
      "the accessor");
    std::size_t size = 1;
    for (std::int32_t i = 0; i < Dim; ++i) {
      size *= shape_[i];
    }

    // BUGBUG HACKHACK. This isn't actually 100% correct.
    // Come back and rethink the whole elementwise_accessor thing.
    auto handle = std::get<0>(other.spans_).data_handle();
    for (std::size_t i = 0; i < size; ++i) {
      access(handle, i) = other.access(0, i);
    }
    return *this;
  }

  LEGATE_HOST_DEVICE [[nodiscard]] reference access(data_handle_type handle,
                                                    std::size_t i) const noexcept
  {
    Point<Dim> p;
    auto offset = handle + i;

    for (auto dim = Dim - 1; dim >= 0; --dim) {
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
  template <typename Function, typename... InputSpans>
  LEGATE_HOST_DEVICE static void elementwise(mdspan_accessor& self, const void* elementwise_obj)
  {
    self = *static_cast<const elementwise_accessor<Function, InputSpans...>*>(elementwise_obj);
  }

  PhysicalStore store_{};
  Point<Dim> shape_{};
  Point<Dim> origin_{};
  accessor_type accessor_{};
  void (*elementwise_)(mdspan_accessor&, const void*){};
  const void* elementwise_obj_{};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Store>
struct value_type_of_
  : meta::if_c<(type_code_of<Store> == Type::Code::NIL), meta::empty, meta::type<Store>> {};

}  // namespace detail

/**
 * @brief An alias for `std::mdspan` with a custom accessor that allows
 *       elementwise access to a `legate::PhysicalStore`.
 *
 * @tparam ElementType The element type of the `mdspan`.
 * @tparam Dim The dimensionality of the `mdspan`.
 */
template <typename ElementType, std::int32_t Dim>
using mdspan_t =  //
  std::mdspan<ElementType,
              std::dextents<coord_t, Dim>,
              std::layout_right,
              detail::mdspan_accessor<ElementType, Dim>>;

template <typename Op, std::int32_t Dim, bool Exclusive = false>
using mdspan_reduction_t =  //
  std::mdspan<
    typename Op::RHS,
    std::dextents<coord_t, Dim>,
    std::layout_right,
    detail::mdspan_accessor<typename Op::RHS, Dim, detail::reduction_accessor<Op, Exclusive>>>;

}  // namespace legate::stl

#include "suffix.hpp"
