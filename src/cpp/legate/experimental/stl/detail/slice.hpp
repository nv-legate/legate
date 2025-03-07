/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/get_logical_store.hpp>
#include <legate/experimental/stl/detail/iterator.hpp>
#include <legate/experimental/stl/detail/mdspan.hpp>
#include <legate/experimental/stl/detail/ranges.hpp>
#include <legate/experimental/stl/detail/stlfwd.hpp>
#include <legate/utilities/assert.h>

#include <numeric>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

namespace detail {

class BroadcastConstraint {
 public:
  legate::tuple<std::uint32_t> axes{};

  [[nodiscard]] auto operator()(legate::Variable self) const
  {
    return legate::broadcast(std::move(self), axes);
  }
};

template <std::size_t Index, std::int32_t... ProjDims, typename Cursor, typename Extents>
LEGATE_HOST_DEVICE [[nodiscard]] auto project_dimension(Cursor cursor,
                                                        const Extents& extents) noexcept
{
  if constexpr (((Index != ProjDims) && ...)) {
    static_cast<void>(cursor);
    static_cast<void>(extents);
    return std::full_extent;
  } else {
    for (auto i : {ProjDims...}) {
      if (i == Index) {
        return cursor % extents.extent(i);
      }
      cursor /= extents.extent(i);
    }
    LEGATE_UNREACHABLE();
  }
}

template <typename Map>
class View {
 public:
  LEGATE_HOST_DEVICE explicit View(Map map) : map_{std::move(map)} {}

  LEGATE_HOST_DEVICE [[nodiscard]] iterator<Map> begin() const { return {map_, map_.begin()}; }

  LEGATE_HOST_DEVICE [[nodiscard]] iterator<Map> end() const { return {map_, map_.end()}; }

 private:
  Map map_{};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <std::int32_t... ProjDims>
class ProjectionPolicy {
 public:
  static_assert(sizeof...(ProjDims) > 0);
  static_assert(((ProjDims >= 0) && ...));

  template <typename ElementType, std::int32_t Dim>
  class Policy {
   public:
    static_assert(sizeof...(ProjDims) < Dim);
    static_assert(((ProjDims < Dim) && ...));

    [[nodiscard]] static LogicalStore aligned_promote(const LogicalStore& from, LogicalStore to)
    {
      LEGATE_ASSERT(from.dim() == Dim);
      LEGATE_ASSERT(from.dim() == to.dim() + sizeof...(ProjDims));

      const Shape shape = from.extents();

      // handle 0D stores specially until legate scalar stores are 0D themselves
      if (to.dim() == 1 && to.volume() == 1) {  //
        to = to.project(0, 0);
      }

      for (auto dim : {ProjDims...}) {  //
        to = to.promote(dim, shape[dim]);
      }

      LEGATE_ASSERT(from.extents() == to.extents());
      return to;
    }

    template <typename OtherElementTypeT, std::int32_t OtherDim>
    using rebind = Policy<OtherElementTypeT, OtherDim>;

    template <typename Mdspan>
    class PhysicalMap : public affine_map<std::int64_t> {
     public:
      static_assert(Mdspan::extents_type::rank() == Dim);
      static_assert(std::is_same_v<typename Mdspan::value_type, ElementType>);

      PhysicalMap() = default;

      LEGATE_HOST_DEVICE explicit PhysicalMap(Mdspan span) : span_{std::move(span)} {}

      template <std::size_t... Is>
      LEGATE_HOST_DEVICE [[nodiscard]] static auto read_impl(std::index_sequence<Is...>,
                                                             const Mdspan& span,
                                                             cursor cursor)
      {
        auto extents = span.extents();
        return std::submdspan(span, project_dimension<Is, ProjDims...>(cursor, extents)...);
      }

      LEGATE_HOST_DEVICE [[nodiscard]] decltype(auto) read(cursor cur) const
      {
        return read_impl(Indices{}, span_, cur);
      }

      LEGATE_HOST_DEVICE [[nodiscard]] cursor end() const
      {
        return (span_.extents().extent(ProjDims) * ... * 1);
      }

      [[nodiscard]] std::array<coord_t, Dim - sizeof...(ProjDims)> shape() const
      {
        std::array<coord_t, Dim - sizeof...(ProjDims)> result;
        for (std::int32_t i = 0, j = 0; i < Dim; ++i) {  //
          if (((i != ProjDims) && ...)) {                //
            result[j++] = span_.extents().extent(i);
          }
        }
        return result;
      }

      Mdspan span_{};
      using Indices    = std::make_index_sequence<Dim>;
      using value_type = decltype(PhysicalMap::read_impl(Indices(), {}, 0));
    };

    class LogicalMap : public affine_map<std::int64_t> {
     public:
      using value_type = logical_store<std::remove_cv_t<ElementType>, Dim - sizeof...(ProjDims)>;

      explicit LogicalMap(LogicalStore store) : store_{std::move(store)}
      {
        LEGATE_ASSERT(store_.dim() == Dim);
      }

      [[nodiscard]] value_type read(cursor cur) const
      {
        auto store = store_;
        int offset = 0;
        for (auto i : {ProjDims...}) {
          auto extent = store.extents()[i - offset];
          store       = store.project(i - offset, cur % extent);
          cur /= extent;
          ++offset;
        }
        return as_typed<std::remove_cv_t<ElementType>, Dim - sizeof...(ProjDims)>(store);
      }

      [[nodiscard]] cursor end() const { return (store_.extents()[ProjDims] * ... * 1); }

      [[nodiscard]] std::array<coord_t, Dim - sizeof...(ProjDims)> shape() const
      {
        std::array<coord_t, Dim - sizeof...(ProjDims)> result;
        for (std::int32_t i = 0, j = 0; i < Dim; ++i) {  //
          if (((i != ProjDims) && ...)) {                //
            result[j++] = store_.extents()[i];
          }
        }
        return result;
      }

     private:
      LogicalStore store_;
    };

    [[nodiscard]] static View<LogicalMap> logical_view(LogicalStore store)
    {
      return View{LogicalMap{std::move(store)}};
    }

    template <typename T, typename E, typename L, typename A>
      requires(std::is_same_v<T const, ElementType const>)
    LEGATE_HOST_DEVICE [[nodiscard]] static View<PhysicalMap<std::mdspan<T, E, L, A>>>
    physical_view(std::mdspan<T, E, L, A> span)
    {
      static_assert(Dim == E::rank());
      return View{PhysicalMap<std::mdspan<T, E, L, A>>(std::move(span))};
    }

    LEGATE_HOST_DEVICE [[nodiscard]] static coord_t size(const LogicalStore& store)
    {
      auto&& shape = store.extents();
      return (shape[ProjDims] * ... * 1);
    }

    LEGATE_HOST_DEVICE [[nodiscard]] static coord_t size(const PhysicalStore& store)
    {
      auto&& shape = store.shape<Dim>();
      return ((shape.hi[ProjDims] - shape.lo[ProjDims]) * ... * 1);
    }

    // TODO(ericniebler): maybe cast this into the mold of a segmented range?
    [[nodiscard]] static std::tuple<BroadcastConstraint> partition_constraints(iteration_kind)
    {
      std::vector<std::uint32_t> axes(Dim);

      std::iota(axes.begin(), axes.end(), 0);
      constexpr std::uint32_t proj_dims[] = {ProjDims...};
      axes.erase(
        std::set_difference(
          axes.begin(), axes.end(), std::begin(proj_dims), std::end(proj_dims), axes.begin()),
        axes.end());
      return std::make_tuple(BroadcastConstraint{tuple<std::uint32_t>{std::move(axes)}});
    }

    [[nodiscard]] static std::tuple<> partition_constraints(reduction_kind) { return {}; }
  };

  template <typename ElementType, std::int32_t Dim>
  using rebind = Policy<ElementType, Dim>;
};

using RowPolicy    = ProjectionPolicy<0>;
using ColumnPolicy = ProjectionPolicy<1>;

class ElementPolicy {
 public:
  template <typename ElementType, std::int32_t Dim>
  class Policy {
   public:
    template <typename OtherElementTypeT, std::int32_t OtherDim>
    using rebind = Policy<OtherElementTypeT, OtherDim>;

    [[nodiscard]] static LogicalStore aligned_promote(const LogicalStore& from, LogicalStore to)
    {
      LEGATE_ASSERT(from.dim() == Dim);
      LEGATE_ASSERT(to.dim() == 1 && to.volume() == 1);

      to = to.project(0, 0);

      auto&& shape = from.extents();
      LEGATE_ASSERT(shape.size() == Dim);
      for (std::int32_t dim = 0; dim < Dim; ++dim) {
        to = to.promote(dim, shape[dim]);
      }
      return to;
    }

    template <typename Mdspan>
    class PhysicalMap : public affine_map<std::int64_t> {
     public:
      using value_type = typename Mdspan::value_type;
      using reference  = typename Mdspan::reference;

      static_assert(Mdspan::extents_type::rank() == Dim);
      static_assert(std::is_same_v<typename Mdspan::value_type, ElementType>);

      PhysicalMap() = default;

      LEGATE_HOST_DEVICE explicit PhysicalMap(Mdspan span) : span_{std::move(span)} {}

      LEGATE_HOST_DEVICE [[nodiscard]] reference read(cursor cur) const
      {
        std::array<coord_t, Dim> p;
        for (std::int32_t i = Dim - 1; i >= 0; --i) {
          p[i] = cur % span_.extents().extent(i);
          cur /= span_.extents().extent(i);
        }
        return span_[p];
      }

      LEGATE_HOST_DEVICE [[nodiscard]] cursor end() const
      {
        cursor result = 1;
        for (std::int32_t i = 0; i < Dim; ++i) {
          result *= span_.extents().extent(i);
        }
        return result;
      }

      [[nodiscard]] std::array<coord_t, Dim> shape() const
      {
        std::array<coord_t, Dim> result;
        for (std::int32_t i = 0; i < Dim; ++i) {
          result[i] = span_.extents().extent(i);
        }
        return result;
      }

      Mdspan span_{};
    };

    [[nodiscard]] static View<PhysicalMap<mdspan_t<ElementType, Dim>>> logical_view(
      const LogicalStore& store)
    {
      return physical_view(as_mdspan<ElementType, Dim>(store));
    }

    template <typename T, typename E, typename L, typename A>
      requires(std::is_same_v<T const, ElementType const>)
    LEGATE_HOST_DEVICE [[nodiscard]] static View<PhysicalMap<std::mdspan<T, E, L, A>>>
    physical_view(std::mdspan<T, E, L, A> span)
    {
      static_assert(Dim == E::rank());
      return View{PhysicalMap<std::mdspan<T, E, L, A>>{std::move(span)}};
    }

    LEGATE_HOST_DEVICE [[nodiscard]] static coord_t size(const LogicalStore& store)
    {
      return static_cast<coord_t>(store.volume());
    }

    LEGATE_HOST_DEVICE [[nodiscard]] static coord_t size(const PhysicalStore& store)
    {
      return store.shape<Dim>().volume();
    }

    [[nodiscard]] static std::tuple<> partition_constraints(ignore) { return {}; }
  };

  template <typename ElementType, std::int32_t Dim>
  using rebind = Policy<ElementType, Dim>;
};

template <typename Policy, typename ElementType, std::int32_t Dim>
using RebindPolicy = typename Policy::template rebind<ElementType, Dim>;

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ElementType, std::int32_t Dim, typename SlicePolicy>
class SliceView {
 public:
  using policy = detail::RebindPolicy<SlicePolicy, ElementType, Dim>;

  explicit SliceView(LogicalStore store) : store_{std::move(store)} {}

  [[nodiscard]] static constexpr std::int32_t dim() noexcept { return Dim; }

  [[nodiscard]] auto begin() const { return policy{}.logical_view(store_).begin(); }

  [[nodiscard]] auto end() const { return policy{}.logical_view(store_).end(); }

  [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>(end() - begin()); }

  [[nodiscard]] logical_store<std::remove_cv_t<ElementType>, Dim> base() const
  {
    return stl::as_typed<std::remove_cv_t<ElementType>, Dim>(store_);
  }

 private:
  [[nodiscard]] friend LogicalStore get_logical_store(const SliceView& slice)
  {
    return slice.store_;
  }

  mutable LogicalStore store_;
};

template <typename ElementType, std::int32_t Dim, typename SlicePolicy>
using slice_view_t = SliceView<ElementType, Dim, RebindPolicy<SlicePolicy, ElementType, Dim>>;

template <typename Store, std::int32_t... ProjDim>
class ProjectionView {
 public:
  using type = slice_view_t<value_type_of_t<Store>, dim_of_v<Store>, ProjectionPolicy<ProjDim...>>;
};

}  // namespace detail

/**
 * @brief A policy for use with `legate::experimental::stl::slice_view` that
 * creates a flat view of all the elements of the store.
 * @ingroup stl-views
 */
using element_policy = detail::ElementPolicy;  // NOLINT(readability-identifier-naming)

/**
 * @brief A policy for use with `legate::experimental::stl::slice_view` that
 * slices a logical store along the row (0th) dimension.
 * @note The elements of the resulting range are logical stores with one fewer
 * dimension than the original store.
 * @ingroup stl-views
 */
using row_policy = detail::RowPolicy;  // NOLINT(readability-identifier-naming)

/**
 * @brief A policy for use with `legate::experimental::stl::slice_view` that
 * slices a logical store along the column (1st) dimension.
 * @note The elements of the resulting range are logical stores with one fewer
 * dimension than the original store.
 * @ingroup stl-views
 */
using column_policy = detail::ColumnPolicy;  // NOLINT(readability-identifier-naming)

/**
 * @brief A policy for use with `legate::experimental::stl::slice_view` that
 * slices a logical store along `ProjDims...` dimensions.
 * @note The elements of the resulting range are logical stores with \em N fewer
 * dimensions than the original store, where \em N is `sizeof...(ProjDims)`.
 * @ingroup stl-views
 */
template <std::int32_t... ProjDims>
using projection_policy =  // NOLINT(readability-identifier-naming)
  detail::ProjectionPolicy<ProjDims...>;

/**
 * @brief A view of a logical store, sliced along some specified dimension(s),
 * resulting in a 1-dimensional range of logical stores.
 *
 * @tparam ElementType The element type of the underlying logical store.
 * @tparam Dim The dimensionality of the underlying logical store.
 * @tparam SlicePolicy A type that determines how the logical store is sliced
 * into a range. Choices include `element_policy`, `row_policy`,
 * `column_policy`, and `projection_policy`.
 *
 * @ingroup stl-views
 */
template <typename ElementType, std::int32_t Dim, typename SlicePolicy>
using slice_view =  // NOLINT(readability-identifier-naming)
  detail::slice_view_t<ElementType, Dim, SlicePolicy>;

template <typename Store>                  //
  requires(logical_store_like<Store>)      //
[[nodiscard]] auto rows_of(Store&& store)  //
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, row_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, row_policy>(
    detail::get_logical_store(std::forward<Store>(store)));
}

template <typename Store>              //
  requires(logical_store_like<Store>)  //
[[nodiscard]] auto columns_of(Store&& store)
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, column_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, column_policy>(
    detail::get_logical_store(std::forward<Store>(store)));
}

template <std::int32_t... ProjDims, typename Store>  //
  requires(logical_store_like<Store>)                //
[[nodiscard]] auto projections_of(Store&& store)
  //-> slice_view<value_type_of_t<Store>, dim_of_v<Store>, projection_policy<ProjDims...>> {
  -> typename detail::ProjectionView<Store, ProjDims...>::type
{
  static_assert((((ProjDims >= 0) && (ProjDims < dim_of_v<Store>)) && ...));
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, projection_policy<ProjDims...>>(
    detail::get_logical_store(std::forward<Store>(store)));
}

template <typename Store>              //
  requires(logical_store_like<Store>)  //
[[nodiscard]] auto elements_of(Store&& store)
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, element_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, element_policy>(
    detail::get_logical_store(std::forward<Store>(store)));
}

namespace detail {

template <typename ElementType>
struct ValueTypeOf;

template <typename ElementType, std::int32_t Dim, typename SlicePolicy>
struct ValueTypeOf<SliceView<ElementType, Dim, SlicePolicy>> {
  using type = ElementType;
};

}  // namespace detail

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
