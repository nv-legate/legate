/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/stlfwd.hpp>
/////////////////
#include <legate.h>

#include <legate/experimental/stl/detail/get_logical_store.hpp>
#include <legate/experimental/stl/detail/mdspan.hpp>
#include <legate/experimental/stl/detail/slice.hpp>
#include <legate/experimental/stl/detail/span.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

/** @cond */
template <typename ElementType, std::int32_t Dim>
class logical_store;

namespace detail {

template <typename ElementType, std::int32_t Dim>
struct ValueTypeOf<logical_store<ElementType, Dim>> {
  using type = ElementType;
};

struct CtorTag {};

/// @c InitListTraits is a helper class that provides a way to extract the shape of a
/// multi-dimensional initializer list and fill a multi-dimensional store with its values.
template <class ElementType, std::int32_t Dim>
struct InitListTraits {
  using type = std::initializer_list<typename InitListTraits<ElementType, Dim - 1>::type>;

  /// Get the shape described by the initializer_list.
  template <std::size_t N = Dim>
  static std::array<std::size_t, N> get_shape(type il, std::array<std::size_t, N> shape = {})
  {
    if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG) && Dim > 1) {
      // Make sure all the inner lists have the same size:
      const std::size_t size = (*il.begin()).size();
      for (const auto& inner : il) {
        LEGATE_ASSERT(inner.size() == size);
      }
    }

    shape[N - Dim] = il.size();
    return InitListTraits<ElementType, Dim - 1>::get_shape(*il.begin(), shape);
  }

  /// Fill the store with the values from the initializer_list.
  template <std::size_t... Indices, std::size_t N, std::int32_t N2>
  static void fill(std::index_sequence<Indices...> indices,
                   std::array<std::size_t, N>& shape,
                   type il,
                   const mdspan_t<ElementType, N2>& span)
  {
    static_assert(sizeof...(Indices) == N && (N2 == N));
    for (const auto& inner : il) {
      InitListTraits<ElementType, Dim - 1>::fill(indices, shape, inner, span);
      ++shape[N - Dim];
    }
    shape[N - Dim] = 0;
  }
};

template <class ElementType>
struct InitListTraits<ElementType, 0> {
  using type = ElementType;

  template <std::size_t N = 0>
  static std::array<std::size_t, N> get_shape(type, std::array<std::size_t, N> shape = {}) noexcept
  {
    return shape;
  }

  template <std::size_t... Indices, std::size_t N, std::int32_t N2>
  static void fill(std::index_sequence<Indices...>,
                   std::array<std::size_t, N>& shape,
                   type value,
                   const mdspan_t<ElementType, N2>& span) noexcept
  {
    static_assert(sizeof...(Indices) == N && (N2 == N));
    span(shape[Indices]...) = value;
  }
};

}  // namespace detail
/** @endcond */

/**
 * @brief A multi-dimensional data container
 *
 * `logical_store` is a multi-dimensional data container for fixed-size elements. Stores are
 * internally partitioned and distributed across the system, but users need not specify the
 * partitioning and distribution directly. Instead, Legate.STL automatically partitions and
 * distributes stores based on how the `logical_store` object is used.
 *
 * `logical_store` objects are value-like and are move-only. They are not immediately associated
 * with a physical allocation. To access the data directly, users must ask for a view into
 * the logical store. This will trigger the allocation of a physical store and the population
 * of the physical memory, which is a blocking operation.
 *
 * @tparam ElementType The type of the elements of the store. The element type must be fixed
 *         size, and it must be one of the
 *         @verbatim embed:rst:inline :ref:`allowable Legate element types <element-types>`.
           @endverbatim
 * @tparam Dim The number of dimensions in the logical store
 *
 * @ingroup stl-containers
 */
template <typename ElementType, std::int32_t Dim>
class logical_store
#if !LEGATE_DEFINED(LEGATE_DOXYGEN)
  : private legate::LogicalStore
#endif
{
  using init_list_traits_t = detail::InitListTraits<ElementType, Dim>;
  using init_list_t        = typename init_list_traits_t::type;

 public:
  static_assert(
    type_code_of_v<ElementType> != legate::Type::Code::NIL,
    "The type of a logical_store<> must be a type that is valid for legate::LogicalStore.");
  using value_type = ElementType;
  // By default, the algorithms treat stores as element-wise.
  using policy = detail::ElementPolicy::Policy<ElementType, Dim>;

  logical_store() = delete;

  /**
   * @brief Create a logical store of a given shape.
   *
   * @tparam Rank The number of dimensions in the store.
   * @param extents The extents of the store.
   *
   * @code
   * logical_store<int, 3> data{{ 10, 20, 30 }};
   * @endcode
   */
  template <std::size_t Rank>
    requires(Rank == Dim)
  explicit logical_store(const std::size_t (&extents)[Rank])
    : LogicalStore{logical_store::create_(extents)}
  {
  }

  /**
   * @overload
   */
  explicit logical_store(std::span<const std::size_t, Dim> extents)
    : LogicalStore{logical_store::create_(std::move(extents))}
  {
  }

  /**
   * @brief Create a logical store of a given shape and fills it with a
   *        given value.
   *
   * @tparam Rank The number of dimensions in the store.
   * @param extents The extents of the store.
   * @param value The value to fill the store with.
   *
   * @code
   * logical_store<int, 3> data{{ 10, 20, 30 }, 42};
   * @endcode
   */
  template <std::size_t Rank>
    requires(Rank == Dim)
  explicit logical_store(const std::size_t (&extents)[Rank], ElementType value)
    : LogicalStore{logical_store::create_(extents)}
  {
    legate::Runtime::get_runtime()->issue_fill(*this, Scalar{std::move(value)});
  }

  /**
   * @overload
   */
  explicit logical_store(std::span<const std::size_t, Dim> extents, ElementType value)
    : LogicalStore{logical_store::create_(std::move(extents))}
  {
    legate::Runtime::get_runtime()->issue_fill(*this, Scalar{std::move(value)});
  }

  /**
   * @brief Create a logical store from a @c std::initializer_list.
   *
   * For stores with two or more dimensions, @c logical_store objects
   * can be initialized from an @c initializer_list as follows:
   *
   * @code {.cpp}
   * logical_store<int, 2> data = { {1, 2, 3}, {4, 5, 6} };
   * @endcode
   *
   * For 1-dimensional stores, however, this syntax can get confused
   * with the constructor that takes the shape of the store as a list
   * of integers. So for 1-D stores, the first argument must be
   * @c std::in_place as shown below:
   *
   * @code {.cpp}
   * logical_store<int, 1> data{std::in_place, {1, 2, 3, 4, 5, 6}};
   * @endcode
   *
   * @param il The initializer list to create the store from. The type
   *          of @c il is a nested @c std::initializer_list of the store's
   *          element type.
   *
   * @pre @li The initializer list must have the same dimensionality (as
   *          determined by list nesting depth) as the store.
   *
   * @par Example:
   * @snippet{trimleft} experimental/stl/store.cc 2D initializer_list
   */
  logical_store(std::in_place_t, init_list_t il)
    : LogicalStore{logical_store::create_(init_list_traits_t::get_shape(il))}
  {
    std::array<std::size_t, Dim> shape{};  // default-initialized to zeros
    const mdspan_t<ElementType, Dim> span = as_mdspan(*this);
    init_list_traits_t::fill(std::make_index_sequence<Dim>{}, shape, il, span);
  }

#if LEGATE_DEFINED(LEGATE_DOXYGEN)
  /**
   * @overload
   *
   * @note This constructor is only available when the dimensionality of the
   * store is greater than 1.
   */
  logical_store(init_list_t il);
#endif

  // NOLINTBEGIN(google-explicit-constructor)
  template <std::int32_t Rank = Dim>
    requires(Rank != 1)
  logical_store(init_list_t il, std::enable_if_t<Rank != 1, int> = 0)
    : LogicalStore{logical_store::create_(init_list_traits_t::get_shape(il))}
  {
    std::array<std::size_t, Dim> shape{};  // default-initialized to zeros
    const mdspan_t<ElementType, Dim> span = as_mdspan(*this);
    init_list_traits_t::fill(std::make_index_sequence<Dim>{}, shape, il, span);
  }
  // NOLINTEND(google-explicit-constructor)

  /**
   * @brief `logical_store` is a move-only type.
   */
  logical_store(logical_store&&)            = default;
  logical_store& operator=(logical_store&&) = default;

  /**
   * @brief Get the dimensionality of the store.
   *
   * @return `std::int32_t` - The number of dimensions in the store.
   */
  [[nodiscard]] static constexpr std::int32_t dim() noexcept { return Dim; }

  /**
   * @brief Retrieve the extents of the store as a `std::array`
   *
   * @return `std::array<std::size_t, Dim>` - The extents of the store.
   */
  [[nodiscard]] std::array<std::size_t, Dim> extents() const noexcept
  {
    auto&& extents = LogicalStore::extents();
    std::array<std::size_t, Dim> result;

    LEGATE_ASSERT(extents.size() == Dim);
    std::copy(&extents[0], &extents[0] + Dim, result.begin());
    return result;
  }

 private:
  template <typename, std::int32_t>
  friend class logical_store;

  [[nodiscard]] static LogicalStore create_(std::span<const std::size_t, Dim> exts)
  {
    Runtime* runtime = legate::Runtime::get_runtime();
    // create_store() takes const-ref for now, but may not always be the case
    // NOLINTNEXTLINE(misc-const-correctness)
    Shape shape({exts.begin(), exts.end()});
    return runtime->create_store(std::move(shape), primitive_type(type_code_of_v<ElementType>));
  }

  [[nodiscard]] static LogicalStore create_(const std::array<std::size_t, Dim> exts)
  {
    return create_(std::span{exts});
  }

  static void validate_(const LogicalStore& store)
  {
    static_assert(sizeof(logical_store) == sizeof(LogicalStore));
    LEGATE_ASSERT(store.type().code() == type_code_of_v<ElementType>);
    LEGATE_ASSERT(store.dim() == Dim || (Dim == 0 && store.dim() == 1));
  }

  logical_store(detail::CtorTag, LogicalStore&& store) : LogicalStore{std::move(store)}
  {
    validate_(*this);
  }

  logical_store(detail::CtorTag, const LogicalStore& store) : LogicalStore{store}
  {
    validate_(*this);
  }

  friend logical_store<ElementType, Dim> as_typed<>(const LogicalStore& store);

  friend LogicalStore get_logical_store(const logical_store& self) noexcept { return self; }

  friend auto as_range(logical_store& self) noexcept { return elements_of(self); }

  friend auto as_range(const logical_store& self) noexcept { return elements_of(self); }
};

/** @cond */
// A specialization for 0-dimensional (scalar) stores:
template <typename ElementType>
class logical_store<ElementType, 0> : private LogicalStore {
 public:
  using value_type = ElementType;
  // By default, the algorithms treat stores as element-wise.
  using policy = detail::ElementPolicy::Policy<ElementType, 0>;

  logical_store() = delete;

  explicit logical_store(std::span<const std::size_t, 0>) : LogicalStore{logical_store::create_()}
  {
  }

  explicit logical_store(std::span<const std::size_t, 0>, ElementType elem)
    : LogicalStore{logical_store::create_(std::move(elem))}
  {
  }

  explicit logical_store(ElementType elem) : LogicalStore{logical_store::create_(std::move(elem))}
  {
  }

  // Make logical_store a move-only type:
  logical_store(logical_store&&)            = default;
  logical_store& operator=(logical_store&&) = default;

  [[nodiscard]] static constexpr std::int32_t dim() noexcept { return 0; }

  [[nodiscard]] std::array<std::size_t, 0> extents() const { return {}; }

 private:
  template <typename, std::int32_t>
  friend class logical_store;

  [[nodiscard]] static LogicalStore create_(ElementType elem = {})
  {
    return legate::Runtime::get_runtime()->create_store(Scalar{std::move(elem)});
  }

  logical_store(detail::CtorTag, LogicalStore&& store) : LogicalStore{std::move(store)} {}

  logical_store(detail::CtorTag, const LogicalStore& store) : LogicalStore{store} {}

  friend logical_store<ElementType, 0> as_typed<>(const LogicalStore& store);

  friend LogicalStore get_logical_store(const logical_store& self) noexcept { return self; }

  friend auto as_range(logical_store& self) noexcept { return elements_of(self); }

  friend auto as_range(const logical_store& self) noexcept { return elements_of(self); }
};
/** @endcond */

/*-***********************************************************************************************
 * Deduction guides for logical_store<>:
 */
template <typename ElementType>
logical_store(std::span<const std::size_t, 0>, ElementType) -> logical_store<ElementType, 0>;

template <typename ElementType, std::size_t Dim>
logical_store(const std::size_t (&)[Dim], ElementType) -> logical_store<ElementType, Dim>;

template <typename ElementType, std::size_t Dim>
logical_store(std::array<std::size_t, Dim>, ElementType) -> logical_store<ElementType, Dim>;

/**
 * @brief Given an untyped `legate::LogicalStore`, return a strongly-typed
 *        `legate::experimental::stl::logical_store`.
 *
 * @tparam ElementType The element type of the `LogicalStore`.
 * @tparam Dim The dimensionality of the `LogicalStore`.
 * @param store The `LogicalStore` to convert.
 * @return `logical_store<ElementType, Dim>`
 * @pre The element type of the `LogicalStore` must be the same as `ElementType`,
 *      and the dimensionality of the `LogicalStore` must be the same as `Dim`.
 *
 * @ingroup stl-containers
 */
template <typename ElementType, std::int32_t Dim>
[[nodiscard]] logical_store<ElementType, Dim> as_typed(const legate::LogicalStore& store)
{
  return {detail::CtorTag{}, store};
}

/** @cond */
namespace detail {

template <std::int32_t Dim, typename Rect>
[[nodiscard]] inline std::array<coord_t, Dim> dynamic_extents(const Rect& shape)
{
  if constexpr (Dim == 0) {
    return {};
  } else {
    const Point<Dim> extents = Point<Dim>::ONES() + shape.hi - shape.lo;
    std::array<coord_t, Dim> result;
    for (std::int32_t i = 0; i < Dim; ++i) {  //
      result[i] = extents[i];
    }
    return result;
  }
}

template <std::int32_t Dim>
[[nodiscard]] inline std::array<coord_t, Dim> dynamic_extents(const legate::PhysicalStore& store)
{
  return dynamic_extents<Dim>(store.shape<Dim ? Dim : 1>());
}
}  // namespace detail
/** @endcond */

/**
 * @brief Given an untyped `legate::PhysicalStore`, return a strongly-typed
 *        `legate::experimental::stl::logical_store`.
 *
 * @tparam ElementType The element type of the `PhysicalStore`.
 * @tparam Dim The dimensionality of the `PhysicalStore`.
 * @param store The `PhysicalStore` to convert.
 * @return `mdspan_t<ElementType, Dim>`
 * @pre The element type of the `PhysicalStore` must be the same as
 *      `ElementType`, and the dimensionality of the `PhysicalStore` must be the
 *      same as `Dim`.
 *
 * @ingroup stl-containers
 */
template <typename ElementType, std::int32_t Dim>
LEGATE_HOST_DEVICE [[nodiscard]] inline mdspan_t<ElementType, Dim> as_mdspan(
  const legate::PhysicalStore& store)
{
  // These can all be *sometimes* moved.
  // NOLINTBEGIN(misc-const-correctness)
  using Mapping = std::layout_right::mapping<std::dextents<coord_t, Dim>>;
  Mapping mapping{detail::dynamic_extents<Dim>(store)};

  using Accessor = detail::MDSpanAccessor<ElementType, Dim>;
  Accessor accessor{store};

  using Handle = typename Accessor::data_handle_type;
  Handle handle{};
  // NOLINTEND(misc-const-correctness)

  return {std::move(handle), std::move(mapping), std::move(accessor)};
}

/**
 * @overload
 */
template <typename ElementType, std::int32_t Dim>
LEGATE_HOST_DEVICE [[nodiscard]] inline mdspan_t<ElementType, Dim> as_mdspan(
  const legate::LogicalStore& store)
{
  return stl::as_mdspan<ElementType, Dim>(store.get_physical_store());
}

/**
 * @overload
 */
template <typename ElementType, std::int32_t Dim>
LEGATE_HOST_DEVICE [[nodiscard]] inline mdspan_t<ElementType, Dim> as_mdspan(
  const legate::PhysicalArray& array)
{
  return stl::as_mdspan<ElementType, Dim>(array.data());
}

/**
 * @overload
 */
template <typename ElementType, std::int32_t Dim, template <typename, std::int32_t> typename StoreT>
  requires(std::is_same_v<logical_store<ElementType, Dim>, StoreT<ElementType, Dim>>)
LEGATE_HOST_DEVICE [[nodiscard]] inline mdspan_t<ElementType, Dim> as_mdspan(
  const StoreT<ElementType, Dim>& store)
{
  return stl::as_mdspan<ElementType, Dim>(get_logical_store(store));
}

/**
 * @overload
 */
template <typename ElementType, typename Extents, typename Layout, typename Accessor>
LEGATE_HOST_DEVICE [[nodiscard]] inline auto as_mdspan(
  std::mdspan<ElementType, Extents, Layout, Accessor> mdspan)
  -> std::mdspan<ElementType, Extents, Layout, Accessor>
{
  return mdspan;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/** @cond */
namespace detail {

template <bool>
class AsMdspanResult {
 public:
  template <typename T, typename ElementType, typename Dim>
  using eval = decltype(stl::as_mdspan<ElementType, Dim::value>(std::declval<T>()));
};

template <>
class AsMdspanResult<true> {
 public:
  template <typename T, typename ElementType, typename Dim>
  using eval = decltype(stl::as_mdspan(std::declval<T>()));
};

}  // namespace detail
/** @endcond */

template <typename T, typename ElementType = void, std::int32_t Dim = -1>
using as_mdspan_t = meta::eval<detail::AsMdspanResult<std::is_void_v<ElementType> && Dim == -1>,
                               T,
                               ElementType,
                               meta::constant<Dim>>;

////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Create an uninitialized zero-dimensional (scalar) logical store of a given type.
 *
 * @tparam ElementType The element type of the resulting `logical_store`.
 * @return `legate::experimental::stl::logical_store<ElementType, 0>`
 *
 * @code {.cpp}
 * auto scalar = stl::create_store<int>({}); // scalar store of type int
 * @endcode
 *
 * @ingroup stl-containers
 */
template <typename ElementType>  //
[[nodiscard]] logical_store<ElementType, 0> create_store(std::span<const std::size_t, 0UL>)
{
  return logical_store<ElementType, 0>{{}};
}

/**
 * @brief Create and initialize a zero-dimensional (scalar) logical store of a given type.
 *
 * @tparam ElementType The element type of the resulting `logical_store`.
 * @return `legate::experimental::stl::logical_store<ElementType, 0>`
 *
 * @code {.cpp}
 * auto scalar1 = stl::create_store({}, 0);      // scalar store of type int
 * // or
 * auto scalar2 = stl::create_store<int>({}, 0); // same thing
 * @endcode
 *
 * @ingroup stl-containers
 */
template <typename ElementType>  //
[[nodiscard]] logical_store<ElementType, 0> create_store(std::span<const std::size_t, 0UL>,
                                                         ElementType value)
{
  return logical_store<ElementType, 0>{{}, std::move(value)};
}

/**
 * @brief Create an uninitialized logical store of a given type and dimensionality.
 *
 * @tparam ElementType The element type of the resulting `logical_store`.
 * @tparam Dim The dimensionality of the resulting `logical_store`.
 * @return `legate::experimental::stl::logical_store<ElementType, Dim>`
 *
 * @code {.cpp}
 * auto store = stl::create_store<int>({40,50,60}); // A 3-dimensional store of type int
 * @endcode
 *
 * @ingroup stl-containers
 */
template <typename ElementType, std::size_t Dim>  //
[[nodiscard]] logical_store<ElementType, Dim> create_store(const std::size_t (&exts)[Dim])
{
  return logical_store<ElementType, static_cast<std::int32_t>(Dim)>{exts};
}

/**
 * @overload
 *
 * @code {.cpp}
 * stl::legate_store<int, 3> input{...};
 * auto aligned = stl::create_store<int>(source.extents());
 * @endcode
 *
 * @ingroup stl-containers
 */
template <typename ElementType, std::size_t Dim>  //
[[nodiscard]] logical_store<ElementType, Dim> create_store(const std::array<std::size_t, Dim>& exts)
{
  return logical_store<ElementType, Dim>{exts};
}

/**
 * @brief Create and initialize a logical store of a given type and dimensionality.
 *
 * @tparam ElementType The element type of the resulting `logical_store`.
 * @tparam Dim The dimensionality of the resulting `logical_store`.
 * @return `legate::experimental::stl::logical_store<ElementType, Dim>`
 *
 * @code {.cpp}
 * auto store = stl::create_store<int>({40,50,60}, 0); // A 3-dimensional store of type int
 * @endcode
 *
 * @ingroup stl-containers
 */
template <typename ElementType, std::int32_t Dim>  //
[[nodiscard]] logical_store<ElementType, Dim> create_store(const std::size_t (&exts)[Dim],
                                                           ElementType value)
{
  return logical_store<ElementType, Dim>{exts, std::move(value)};
}

/**
 * @overload
 *
 * @code {.cpp}
 * stl::legate_store<int, 3> input{...};
 * auto aligned = stl::create_store<int>(source.extents(), 0);
 * @endcode
 *
 * @ingroup stl-containers
 */
template <typename ElementType, std::size_t Dim>  //
[[nodiscard]] logical_store<ElementType, Dim> create_store(const std::array<std::size_t, Dim>& exts,
                                                           ElementType value)
{
  return logical_store<ElementType, Dim>{exts, std::move(value)};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Create an uninitialized zero-dimensional (scalar) logical store of a given type.
 *
 * @tparam ElementType The element type of the resulting `logical_store`.
 * @return `legate::experimental::stl::logical_store<ElementType, 0>`
 *
 * @code {.cpp}
 * auto scalar = stl::scalar(0); // scalar store of type int
 * @endcode
 *
 * @ingroup stl-containers
 */
template <typename ElementType>
[[nodiscard]] logical_store<ElementType, 0> scalar(ElementType value)
{
  return logical_store<ElementType, 0>{{}, std::move(value)};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename SlicePolicy, typename ElementType, std::int32_t Dim>
[[nodiscard]] auto slice_as(const logical_store<ElementType, Dim>& store)
{
  return slice_view<ElementType, Dim, SlicePolicy>{get_logical_store(store)};
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
