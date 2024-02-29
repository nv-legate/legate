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
class broadcast_constraint {
 public:
  legate::tuple<std::uint32_t> axes{};

  [[nodiscard]] auto operator()(legate::Variable self) const
  {
    return legate::broadcast(std::move(self), axes);
  }
};

template <std::size_t Index, std::int32_t... ProjDims, typename Cursor, typename Extents>
LEGATE_STL_ATTRIBUTE((host, device))
[[nodiscard]] auto project_dimension(Cursor cursor, const Extents& extents) noexcept
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

template <typename Map>
class view {
 public:
  LEGATE_STL_ATTRIBUTE((host, device))
  explicit view(Map map) : map_{std::move(map)} {}

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] iterator<Map> begin() const { return {map_, map_.begin()}; }

  LEGATE_STL_ATTRIBUTE((host, device))  //
  [[nodiscard]] iterator<Map> end() const { return {map_, map_.end()}; }

 private:
  Map map_{};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <std::int32_t... ProjDims>
class projection_policy {
 public:
  static_assert(sizeof...(ProjDims) > 0);
  static_assert(((ProjDims >= 0) && ...));

  template <typename ElementType, std::int32_t Dim>
  class policy {
   public:
    static_assert(sizeof...(ProjDims) < Dim);
    static_assert(((ProjDims < Dim) && ...));

    [[nodiscard]] static LogicalStore aligned_promote(const LogicalStore& from, LogicalStore to)
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

    template <typename OtherElementTypeT, std::int32_t OtherDim>
    using rebind = policy<OtherElementTypeT, OtherDim>;

    template <typename Mdspan>
    class physical_map : public affine_map<std::int64_t> {
     public:
      static_assert(Mdspan::extents_type::rank() == Dim);
      static_assert(std::is_same_v<typename Mdspan::value_type, ElementType>);

      physical_map() = default;

      LEGATE_STL_ATTRIBUTE((host, device))
      explicit physical_map(Mdspan span) : span_{std::move(span)} {}

      template <std::size_t... Is>
      LEGATE_STL_ATTRIBUTE((host, device))  //
      [[nodiscard]] static auto read_impl(std::index_sequence<Is...>, Mdspan span, cursor cursor)
      {
        auto extents = span.extents();
        return std::submdspan(span, project_dimension<Is, ProjDims...>(cursor, extents)...);
      }

      LEGATE_STL_ATTRIBUTE((host, device))  //
      [[nodiscard]] decltype(auto) read(cursor cur) const
      {
        return read_impl(Indices{}, span_, cur);
      }

      LEGATE_STL_ATTRIBUTE((host, device))  //
      [[nodiscard]] cursor end() const { return (span_.extents().extent(ProjDims) * ... * 1); }

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
      using value_type = decltype(physical_map::read_impl(Indices(), {}, 0));
    };

    class logical_map : public affine_map<std::int64_t> {
     public:
      using value_type = logical_store<std::remove_cv_t<ElementType>, Dim - sizeof...(ProjDims)>;

      logical_map() = default;

      // LEGATE_STL_ATTRIBUTE((host,device))
      explicit logical_map(LogicalStore store) : store_{std::move(store)}
      {
        LegateAssert(store_.dim() == Dim);
      }

      // LEGATE_STL_ATTRIBUTE((host,device))
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

      // LEGATE_STL_ATTRIBUTE((host,device))
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
      LogicalStore store_{};
    };

    // LEGATE_STL_ATTRIBUTE((host,device))
    [[nodiscard]] static view<logical_map> logical_view(LogicalStore store)
    {
      return view{logical_map{std::move(store)}};
    }

    template <typename T, typename E, typename L, typename A>
      requires(std::is_same_v<T const, ElementType const>)
    LEGATE_STL_ATTRIBUTE((host, device))  //
      [[nodiscard]] static view<physical_map<std::mdspan<T, E, L, A>>> physical_view(
        std::mdspan<T, E, L, A> span)
    {
      static_assert(Dim == E::rank());
      return view{physical_map<std::mdspan<T, E, L, A>>(std::move(span))};
    }

    LEGATE_STL_ATTRIBUTE((host, device))  //
    [[nodiscard]] static coord_t size(const LogicalStore& store)
    {
      auto&& shape = store.extents();
      return (shape[ProjDims] * ... * 1);
    }

    LEGATE_STL_ATTRIBUTE((host, device))  //
    [[nodiscard]] static coord_t size(const PhysicalStore& store)
    {
      auto&& shape = store.shape<Dim>();
      return ((shape.hi[ProjDims] - shape.lo[ProjDims]) * ... * 1);
    }

    // TODO(ericniebler): maybe cast this into the mold of a segmented range?
    [[nodiscard]] static std::tuple<broadcast_constraint> partition_constraints(iteration_kind)
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

    [[nodiscard]] static std::tuple<> partition_constraints(reduction_kind) { return {}; }
  };

  template <typename ElementType, std::int32_t Dim>
  using rebind = policy<ElementType, Dim>;
};

using row_policy    = projection_policy<0>;
using column_policy = projection_policy<1>;

class element_policy {
 public:
  template <typename ElementType, std::int32_t Dim>
  class policy {
   public:
    template <typename OtherElementTypeT, std::int32_t OtherDim>
    using rebind = policy<OtherElementTypeT, OtherDim>;

    [[nodiscard]] static LogicalStore aligned_promote(const LogicalStore& from, LogicalStore to)
    {
      LegateAssert(from.dim() == Dim);
      LegateAssert(to.dim() == 1 && to.volume() == 1);

      to = to.project(0, 0);

      auto&& shape = from.extents();
      LegateAssert(shape.size() == Dim);
      for (std::int32_t dim = 0; dim < Dim; ++dim) {
        to = to.promote(dim, shape[dim]);
      }
      return to;
    }

    template <typename Mdspan>
    class physical_map : public affine_map<std::int64_t> {
     public:
      using value_type = typename Mdspan::value_type;
      using reference  = typename Mdspan::reference;

      static_assert(Mdspan::extents_type::rank() == Dim);
      static_assert(std::is_same_v<typename Mdspan::value_type, ElementType>);

      physical_map() = default;

      LEGATE_STL_ATTRIBUTE((host, device))
      explicit physical_map(Mdspan span) : span_{std::move(span)} {}

      LEGATE_STL_ATTRIBUTE((host, device))  //
      [[nodiscard]] reference read(cursor cur) const
      {
        std::array<coord_t, Dim> p;
        for (std::int32_t i = Dim - 1; i >= 0; --i) {
          p[i] = cur % span_.extents().extent(i);
          cur /= span_.extents().extent(i);
        }
        return span_[p];
      }

      LEGATE_STL_ATTRIBUTE((host, device))  //
      [[nodiscard]] cursor end() const
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

    [[nodiscard]] static view<physical_map<mdspan_t<ElementType, Dim>>> logical_view(
      const LogicalStore& store)
    {
      return physical_view(as_mdspan<ElementType, Dim>(store));
    }

    template <typename T, typename E, typename L, typename A>
      requires(std::is_same_v<T const, ElementType const>)
    LEGATE_STL_ATTRIBUTE((host, device))  //
      [[nodiscard]] static view<physical_map<std::mdspan<T, E, L, A>>> physical_view(
        std::mdspan<T, E, L, A> span)
    {
      static_assert(Dim == E::rank());
      return view{physical_map<std::mdspan<T, E, L, A>>{std::move(span)}};
    }

    LEGATE_STL_ATTRIBUTE((host, device))  //
    [[nodiscard]] static coord_t size(const LogicalStore& store)
    {
      return static_cast<coord_t>(store.volume());
    }

    LEGATE_STL_ATTRIBUTE((host, device))  //
    [[nodiscard]] static coord_t size(const PhysicalStore& store)
    {
      return store.shape<Dim>().volume();
    }

    [[nodiscard]] static std::tuple<> partition_constraints(ignore) { return {}; }
  };

  template <typename ElementType, std::int32_t Dim>
  using rebind = policy<ElementType, Dim>;
};

template <typename PolicyT, typename ElementType, std::int32_t Dim>
using rebind_policy = typename PolicyT::template rebind<ElementType, Dim>;

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ElementType, std::int32_t Dim, typename SlicePolicy>
class slice_view {
 public:
  using policy = detail::rebind_policy<SlicePolicy, ElementType, Dim>;

  explicit slice_view(LogicalStore store) : store_{std::move(store)} {}

  [[nodiscard]] static constexpr std::int32_t dim() noexcept { return Dim; }

  [[nodiscard]] auto begin() const { return policy{}.logical_view(store_).begin(); }

  [[nodiscard]] auto end() const { return policy{}.logical_view(store_).end(); }

  [[nodiscard]] std::size_t size() const { return static_cast<std::size_t>(end() - begin()); }

  [[nodiscard]] logical_store<std::remove_cv_t<ElementType>, Dim> base() const
  {
    return stl::as_typed<std::remove_cv_t<ElementType>, Dim>(store_);
  }

 private:
  [[nodiscard]] friend LogicalStore get_logical_store(const slice_view& slice)
  {
    return slice.store_;
  }

  mutable LogicalStore store_{};
};

template <typename ElementType, std::int32_t Dim, typename SlicePolicy>
using slice_view_t = slice_view<ElementType, Dim, rebind_policy<SlicePolicy, ElementType, Dim>>;

template <typename Store, std::int32_t... ProjDim>
class projection_view {
 public:
  using type = slice_view_t<value_type_of_t<Store>, dim_of_v<Store>, projection_policy<ProjDim...>>;
};

}  // namespace detail

template <typename ElementType, std::int32_t Dim, typename SlicePolicy>
using slice_view = detail::slice_view_t<ElementType, Dim, SlicePolicy>;

template <typename Store>                  //
  requires(logical_store_like<Store>)      //
[[nodiscard]] auto rows_of(Store&& store)  //
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::row_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::row_policy>(
    detail::get_logical_store(std::forward<Store>(store)));
}

template <typename Store>              //
  requires(logical_store_like<Store>)  //
[[nodiscard]] auto columns_of(Store&& store)
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::column_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::column_policy>(
    detail::get_logical_store(std::forward<Store>(store)));
}

template <std::int32_t... ProjDims, typename Store>  //
  requires(logical_store_like<Store>)                //
[[nodiscard]] auto projections_of(Store&& store)
  //-> slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::projection_policy<ProjDims...>> {
  -> typename detail::projection_view<Store, ProjDims...>::type
{
  static_assert((((ProjDims >= 0) && (ProjDims < dim_of_v<Store>)) && ...));
  return slice_view<value_type_of_t<Store>,
                    dim_of_v<Store>,
                    detail::projection_policy<ProjDims...>>(
    detail::get_logical_store(std::forward<Store>(store)));
}

template <typename Store>              //
  requires(logical_store_like<Store>)  //
[[nodiscard]] auto elements_of(Store&& store)
  -> slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::element_policy>
{
  return slice_view<value_type_of_t<Store>, dim_of_v<Store>, detail::element_policy>(
    detail::get_logical_store(std::forward<Store>(store)));
}

namespace detail {

template <typename ElementType>
struct value_type_of_;

template <typename ElementType, std::int32_t Dim, typename SlicePolicy>
class value_type_of_<slice_view<ElementType, Dim, SlicePolicy>> {
 public:
  using type = ElementType;
};

}  // namespace detail

}  // namespace legate::stl

#include "suffix.hpp"
