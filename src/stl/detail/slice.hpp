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

#include "get_logical_store.hpp"
#include "iterator.hpp"
#include "mdspan.hpp"
#include "ranges.hpp"
#include "stlfwd.hpp"

#include <numeric>

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

namespace detail {
struct broadcast_constraint {
  legate::tuple<std::uint32_t> axes;

  auto operator()(legate::Variable self) const { return legate::broadcast(self, axes); }
};

template <std::size_t Index, std::int32_t... ProjDims, class Cursor, class Extents>
LEGATE_STL_ATTRIBUTE((host, device))
auto project_dimension(Cursor cursor, Extents extents) noexcept
{
  if constexpr (((Index != ProjDims) && ...)) {
    return std::full_extent;
  } else {
    for (auto i : {ProjDims...}) {
      if (i == Index) {
        return cursor % extents.extent(i);
      }
      cursor /= extents.extent(i);
    }
    LegateUnreachable();
  }
}

template <class Map>
struct view {
  LEGATE_STL_ATTRIBUTE((host, device))
  explicit view(Map map) : map_(std::move(map)) {}

  LEGATE_STL_ATTRIBUTE((host, device))  //
  iterator<Map> begin() const { return iterator(map_, map_.begin()); }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  iterator<Map> end() const { return iterator(map_, map_.end()); }

 private:
  Map map_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <std::int32_t... ProjDims>
struct projection_policy {
  static_assert(sizeof...(ProjDims) > 0);
  static_assert(((ProjDims >= 0) && ...));

  template <class ElementType, std::int32_t Dim>
  struct policy {
    static_assert(sizeof...(ProjDims) < Dim);
    static_assert(((ProjDims < Dim) && ...));

    static LogicalStore aligned_promote(LogicalStore from, LogicalStore to)
    {
      LegateAssert(from.dim() == Dim);
      LegateAssert(from.dim() == to.dim() + sizeof...(ProjDims));

      const Shape shape = from.extents();

      // handle 0D stores specially until legate scalar stores are 0D themselves
      if (to.dim() == 1 && to.volume() == 1) {  //
        to = to.project(0, 0);
      }

      for (auto dim : {ProjDims...}) {  //
        to = to.promote(dim, shape[dim]);
      }

      LegateAssert(from.extents() == to.extents());
      return to;
    }

    template <class OtherElementTypeT, std::int32_t OtherDim>
    using rebind = policy<OtherElementTypeT, OtherDim>;

    template <class Mdspan>
    struct physical_map : affine_map<std::int64_t> {
      static_assert(Mdspan::extents_type::rank() == Dim);
      static_assert(std::is_same_v<typename Mdspan::value_type, ElementType>);

      physical_map() = default;

      LEGATE_STL_ATTRIBUTE((host, device))
      explicit physical_map(Mdspan span) : span_(span) {}

      template <std::size_t... Is>
      LEGATE_STL_ATTRIBUTE((host, device))  //
      static auto read_impl(std::index_sequence<Is...>, Mdspan span, cursor cursor)
      {
        auto extents = span.extents();
        return std::submdspan(span, project_dimension<Is, ProjDims...>(cursor, extents)...);
      }

      LEGATE_STL_ATTRIBUTE((host, device))  //
      decltype(auto) read(cursor cur) const { return read_impl(Indices(), span_, cur); }

      LEGATE_STL_ATTRIBUTE((host, device))  //
      cursor end() const { return (span_.extents().extent(ProjDims) * ... * 1); }

      std::array<coord_t, Dim - sizeof...(ProjDims)> shape() const
      {
        std::array<coord_t, Dim - sizeof...(ProjDims)> result;
        for (std::int32_t i = 0, j = 0; i < Dim; ++i) {  //
          if (((i != ProjDims) && ...)) {                //
            result[j++] = span_.extents().extent(i);
          }
        }
        return result;
      }

      Mdspan span_;
      using Indices    = std::make_index_sequence<Dim>;
      using value_type = decltype(physical_map::read_impl(Indices(), {}, 0));
    };

    struct logical_map : affine_map<std::int64_t> {
      using value_type = logical_store<std::remove_cv_t<ElementType>, Dim - sizeof...(ProjDims)>;

      logical_map() = default;

      // LEGATE_STL_ATTRIBUTE((host,device))
      explicit logical_map(LogicalStore store) : store_(store)
      {
        LegateAssert(store_.dim() == Dim);
      }

      // LEGATE_STL_ATTRIBUTE((host,device))
      value_type read(cursor cur) const
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

      // LEGATE_STL_ATTRIBUTE((host,device))
      cursor end() const { return (store_.extents()[ProjDims] * ... * 1); }

      std::array<coord_t, Dim - sizeof...(ProjDims)> shape() const
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

    // LEGATE_STL_ATTRIBUTE((host,device))
    static view<logical_map> logical_view(const LogicalStore& store)
    {
      return view{logical_map(store)};
    }

    template <class T, class E, class L, class A>
      requires(std::is_same_v<T const, ElementType const>)
    LEGATE_STL_ATTRIBUTE((host, device)) static view<
      physical_map<std::mdspan<T, E, L, A>>> physical_view(std::mdspan<T, E, L, A> span)
    {
      static_assert(Dim == E::rank());
      return view{physical_map<std::mdspan<T, E, L, A>>(span)};
    }

    LEGATE_STL_ATTRIBUTE((host, device))  //
    static coord_t size(const LogicalStore& store)
    {
      const Shape shape = store.extents();
      return (shape[ProjDims] * ... * 1);
    }

    LEGATE_STL_ATTRIBUTE((host, device))  //
    static coord_t size(const PhysicalStore& store)
    {
      const Rect<Dim> shape = store.shape<Dim>();
      return ((shape.hi[ProjDims] - shape.lo[ProjDims]) * ... * 1);
    }

    // TODO: maybe cast this into the mold of a segmented range?
    static std::tuple<broadcast_constraint> partition_constraints(iteration_kind)
    {
      std::vector<std::uint32_t> axes(Dim);
      std::iota(axes.begin(), axes.end(), 0);
      constexpr std::uint32_t proj_dims[] = {ProjDims...};
      axes.erase(
        std::set_difference(
          axes.begin(), axes.end(), std::begin(proj_dims), std::end(proj_dims), axes.begin()),
        axes.end());
      return std::make_tuple(broadcast_constraint{tuple<std::uint32_t>{std::move(axes)}});
    }

    static std::tuple<> partition_constraints(reduction_kind) { return {}; }
  };

  template <class ElementType, std::int32_t Dim>
  using rebind = policy<ElementType, Dim>;
};

using row_policy    = projection_policy<0>;
using column_policy = projection_policy<1>;

struct element_policy {
  template <class ElementType, std::int32_t Dim>
  struct policy {
    template <class OtherElementTypeT, std::int32_t OtherDim>
    using rebind = policy<OtherElementTypeT, OtherDim>;

    static LogicalStore aligned_promote(LogicalStore from, LogicalStore to)
    {
      LegateAssert(from.dim() == Dim);
      LegateAssert(to.dim() == 1 && to.volume() == 1);

      to = to.project(0, 0);

      const Shape shape = from.extents();
      LegateAssert(shape.ndim() == Dim);
      for (std::int32_t dim = 0; dim < Dim; ++dim) {
        to = to.promote(dim, shape[dim]);
      }
      return to;
    }

    template <class Mdspan>
    struct physical_map : affine_map<std::int64_t> {
      using value_type = typename Mdspan::value_type;
      using reference  = typename Mdspan::reference;

      static_assert(Mdspan::extents_type::rank() == Dim);
      static_assert(std::is_same_v<typename Mdspan::value_type, ElementType>);

      physical_map() = default;

      LEGATE_STL_ATTRIBUTE((host, device))
      explicit physical_map(Mdspan span) : span_(span) {}

      LEGATE_STL_ATTRIBUTE((host, device))  //
      reference read(cursor cur) const
      {
        std::array<coord_t, Dim> p;
        for (std::int32_t i = Dim - 1; i >= 0; --i) {
          p[i] = cur % span_.extents().extent(i);
          cur /= span_.extents().extent(i);
        }
        return span_[p];
      }

      LEGATE_STL_ATTRIBUTE((host, device))  //
      cursor end() const
      {
        cursor result = 1;
        for (std::int32_t i = 0; i < Dim; ++i) {
          result *= span_.extents().extent(i);
        }
        return result;
      }

      std::array<coord_t, Dim> shape() const
      {
        std::array<coord_t, Dim> result;
        for (std::int32_t i = 0; i < Dim; ++i) {
          result[i] = span_.extents().extent(i);
        }
        return result;
      }

      Mdspan span_;
    };

    static view<physical_map<mdspan_t<ElementType, Dim>>> logical_view(const LogicalStore& store)
    {
      return physical_view(as_mdspan<ElementType, Dim>(store));
    }

    template <class T, class E, class L, class A>
      requires(std::is_same_v<T const, ElementType const>)
    LEGATE_STL_ATTRIBUTE((host, device)) static view<
      physical_map<std::mdspan<T, E, L, A>>> physical_view(std::mdspan<T, E, L, A> span)
    {
      static_assert(Dim == E::rank());
      return view{physical_map<std::mdspan<T, E, L, A>>(span)};
    }

    LEGATE_STL_ATTRIBUTE((host, device))  //
    static coord_t size(const LogicalStore& store) { return store.volume(); }

    LEGATE_STL_ATTRIBUTE((host, device))  //
    static coord_t size(const PhysicalStore& store) { return store.shape<Dim>().volume(); }

    static std::tuple<> partition_constraints(ignore) { return {}; }
  };

  template <class ElementType, std::int32_t Dim>
  using rebind = policy<ElementType, Dim>;
};

template <class PolicyT, class ElementType, std::int32_t Dim>
using rebind_policy = typename PolicyT::template rebind<ElementType, Dim>;

////////////////////////////////////////////////////////////////////////////////////////////////////
template <class ElementType, std::int32_t Dim, class SlicePolicy>
class slice_view {
 public:
  using policy = detail::rebind_policy<SlicePolicy, ElementType, Dim>;

  explicit slice_view(LogicalStore store) : store_(std::move(store)) {}

  static constexpr std::int32_t dim() noexcept { return Dim; }

  auto begin() const { return policy().logical_view(store_).begin(); }

  auto end() const { return policy().logical_view(store_).end(); }

  std::size_t size() const { return static_cast<std::size_t>(end() - begin()); }

  logical_store<std::remove_cv_t<ElementType>, Dim> base() const
  {
    return stl::as_typed<std::remove_cv_t<ElementType>, Dim>(store_);
  }

 private:
  friend LogicalStore get_logical_store(const slice_view& slice) { return slice.store_; }

  mutable LogicalStore store_;
};

template <class ElementType, std::int32_t Dim, class SlicePolicy>
using slice_view_t = slice_view<ElementType, Dim, rebind_policy<SlicePolicy, ElementType, Dim>>;

template <class Store, std::int32_t... ProjDim>
struct projection_view {
  using type = slice_view_t<value_type_of_t<Store>, dim_of_v<Store>, projection_policy<ProjDim...>>;
};
}  // namespace detail

template <class ElementType, std::int32_t Dim, class SlicePolicy>
using slice_view = detail::slice_view_t<ElementType, Dim, SlicePolicy>;

template <class Store>                 //
  requires(logical_store_like<Store>)  //
auto rows_of(Store&& store)            //
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::row_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::row_policy>(
    detail::get_logical_store(store));
}

template <class Store>                 //
  requires(logical_store_like<Store>)  //
auto columns_of(Store&& store)
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::column_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::column_policy>(
    detail::get_logical_store(store));
}

template <std::int32_t... ProjDims, class Store>  //
  requires(logical_store_like<Store>)             //
auto projections_of(Store&& store)
  //-> slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::projection_policy<ProjDims...>> {
  -> typename detail::projection_view<Store, ProjDims...>::type
{
  static_assert((((ProjDims >= 0) && (ProjDims < dim_of_v<Store>)) && ...));
  return slice_view<value_type_of_t<Store>,
                    dim_of_v<Store>,
                    detail::projection_policy<ProjDims...>>(detail::get_logical_store(store));
}

template <class Store>                 //
  requires(logical_store_like<Store>)  //
auto elements_of(Store&& store)
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::element_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::element_policy>(
    detail::get_logical_store(store));
}

namespace detail {
template <class ElementType>
struct value_type_of_;

template <class ElementType, std::int32_t Dim, class SlicePolicy>
struct value_type_of_<slice_view<ElementType, Dim, SlicePolicy>> {
  using type = ElementType;
};
}  // namespace detail

}  // namespace legate::stl

#include "suffix.hpp"
