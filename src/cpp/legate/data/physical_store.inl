/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/physical_store.h>
#include <legate/utilities/detail/mdspan/util.h>

namespace legate::detail::store_detail {

template <typename ACC, typename T, std::int32_t N>
class RegionInstanceTransAccessorFn {
 public:
  template <std::int32_t M>
  ACC operator()(const Realm::RegionInstance& instance,
                 Realm::FieldID fid,
                 const Legion::AffineTransform<M, N>& transform,
                 const Rect<N>& bounds)
  {
    return {instance,
            fid,
            transform,
            bounds,
            sizeof(T) /* = actual_field_size */,
            false /* = check_field_size */};
  }
};

template <typename ACC, typename T, std::int32_t N>
class RegionInstanceTransReductionAccessorFn {
 public:
  template <std::int32_t M>
  ACC operator()(const Realm::RegionInstance& instance,
                 Realm::FieldID fid,
                 const Legion::AffineTransform<M, N>& transform,
                 const Rect<N>& bounds)
  {
    return {instance,
            fid,
            transform,
            bounds,
            false /* = silence_warnings */,
            nullptr /* = warning_string */,
            0 /* = subfield_offset */,
            sizeof(T) /* = actual_field_size */,
            false /* = check_field_size */};
  }
};

}  // namespace legate::detail::store_detail

namespace legate {

template <typename T, std::int32_t DIM, bool VALIDATE_TYPE>
PhysicalStore::mdspan_type<const T, DIM> PhysicalStore::span_read_accessor(
  std::size_t elem_size) const
{
  return span_read_accessor<T, DIM, VALIDATE_TYPE>(shape<DIM>(), elem_size);
}

template <typename T, std::int32_t DIM, bool VALIDATE_TYPE>
PhysicalStore::mdspan_type<T, DIM> PhysicalStore::span_write_accessor(std::size_t elem_size)
{
  return span_write_accessor<T, DIM, VALIDATE_TYPE>(shape<DIM>(), elem_size);
}

template <typename T, std::int32_t DIM, bool VALIDATE_TYPE>
PhysicalStore::mdspan_type<T, DIM> PhysicalStore::span_read_write_accessor(std::size_t elem_size)
{
  return span_read_write_accessor<T, DIM, VALIDATE_TYPE>(shape<DIM>(), elem_size);
}

template <typename Redop, bool EXCLUSIVE, std::int32_t DIM, bool VALIDATE_TYPE>
PhysicalStore::mdspan_type<typename Redop::LHS, DIM, detail::ReductionAccessor<Redop, EXCLUSIVE>>
PhysicalStore::span_reduce_accessor(std::size_t elem_size)
{
  return span_reduce_accessor<Redop, EXCLUSIVE, DIM, VALIDATE_TYPE>(shape<DIM>(), elem_size);
}

namespace mdspan_detail {

template <typename T,
          typename AccessorPolicy = ::cuda::std::default_accessor<T>,
          typename Acc,
          std::int32_t DIM>
[[nodiscard]] PhysicalStore::mdspan_type<T, DIM, AccessorPolicy> make_accessor(
  const Acc& legion_accessor, const Rect<DIM>& bounds, std::size_t elem_size)
{
  static_assert(DIM >= 0);
  static_assert(DIM <= LEGATE_MAX_DIM);

  auto strides    = ::cuda::std::array<std::size_t, DIM>{};
  auto* const ptr = legion_accessor.ptr(bounds, strides.data(), elem_size);

  using mdspan_type  = PhysicalStore::mdspan_type<T, DIM, AccessorPolicy>;
  using mapping_type = typename mdspan_type::mapping_type;

  return mdspan_type{ptr, mapping_type{detail::mdspan_detail::dynamic_extents(bounds), strides}};
}

}  // namespace mdspan_detail

template <typename T, std::int32_t DIM, bool VALIDATE_TYPE>
PhysicalStore::mdspan_type<const T, DIM> PhysicalStore::span_read_accessor(
  const Rect<DIM>& bounds, std::size_t elem_size) const
{
  return mdspan_detail::make_accessor<const T>(
    read_accessor<T, DIM, VALIDATE_TYPE>(bounds), bounds, elem_size);
}

template <typename T, std::int32_t DIM, bool VALIDATE_TYPE>
PhysicalStore::mdspan_type<T, DIM> PhysicalStore::span_write_accessor(const Rect<DIM>& bounds,
                                                                      std::size_t elem_size)
{
  return mdspan_detail::make_accessor<T>(
    write_accessor<T, DIM, VALIDATE_TYPE>(bounds), bounds, elem_size);
}

template <typename T, std::int32_t DIM, bool VALIDATE_TYPE>
PhysicalStore::mdspan_type<T, DIM> PhysicalStore::span_read_write_accessor(const Rect<DIM>& bounds,
                                                                           std::size_t elem_size)
{
  return mdspan_detail::make_accessor<T>(
    read_write_accessor<T, DIM, VALIDATE_TYPE>(bounds), bounds, elem_size);
}

template <typename Redop, bool EXCLUSIVE, std::int32_t DIM, bool VALIDATE_TYPE>
PhysicalStore::mdspan_type<typename Redop::LHS, DIM, detail::ReductionAccessor<Redop, EXCLUSIVE>>
PhysicalStore::span_reduce_accessor(const Rect<DIM>& bounds, std::size_t elem_size)
{
  return mdspan_detail::make_accessor<typename Redop::LHS,
                                      detail::ReductionAccessor<Redop, EXCLUSIVE>>(
    reduce_accessor<Redop, EXCLUSIVE, DIM, VALIDATE_TYPE>(bounds), bounds, elem_size);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRO<T, DIM> PhysicalStore::read_accessor() const
{
  return read_accessor<T, DIM, VALIDATE_TYPE>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorWO<T, DIM> PhysicalStore::write_accessor() const
{
  return write_accessor<T, DIM, VALIDATE_TYPE>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRW<T, DIM> PhysicalStore::read_write_accessor() const
{
  return read_write_accessor<T, DIM, VALIDATE_TYPE>(shape<DIM>());
}

template <typename OP, bool EXCLUSIVE, int DIM, bool VALIDATE_TYPE>
AccessorRD<OP, EXCLUSIVE, DIM> PhysicalStore::reduce_accessor() const
{
  return reduce_accessor<OP, EXCLUSIVE, DIM, VALIDATE_TYPE>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRO<T, DIM> PhysicalStore::read_accessor(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension_(DIM);
    check_accessor_type_<T>();
  }

  return create_region_instance_accessor_<AccessorRO<T, DIM>, T, DIM>(bounds, VALIDATE_TYPE);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorWO<T, DIM> PhysicalStore::write_accessor(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension_(DIM);
    check_accessor_type_<T>();
  }

  check_write_access_();

  return create_region_instance_accessor_<AccessorWO<T, DIM>, T, DIM>(bounds, VALIDATE_TYPE);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRW<T, DIM> PhysicalStore::read_write_accessor(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension_(DIM);
    check_accessor_type_<T>();
  }

  check_write_access_();

  return create_region_instance_accessor_<AccessorRW<T, DIM>, T, DIM>(bounds, VALIDATE_TYPE);
}

template <typename OP, bool EXCLUSIVE, int DIM, bool VALIDATE_TYPE>
AccessorRD<OP, EXCLUSIVE, DIM> PhysicalStore::reduce_accessor(const Rect<DIM>& bounds) const
{
  using T = typename OP::LHS;
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension_(DIM);
    check_accessor_type_<T>();
  }

  check_reduction_access_();

  return create_region_instance_reduction_accessor_<AccessorRD<OP, EXCLUSIVE, DIM>, T, DIM>(
    bounds, VALIDATE_TYPE);
}

template <typename T, std::int32_t DIM>
Buffer<T, DIM> PhysicalStore::create_output_buffer(const Point<DIM>&, bool) const
{
  return {};
}

template <typename TYPE_CODE>
inline TYPE_CODE PhysicalStore::code() const
{
  return static_cast<TYPE_CODE>(type().code());
}

template <std::int32_t DIM>
Rect<DIM> PhysicalStore::shape() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  check_shape_dimension_(DIM);
  if (dim() > 0) {
    return domain().bounds<DIM, coord_t>();
  }

  auto p = Point<DIM>::ZEROES();
  return {p, p};
}

template <typename VAL>
VAL PhysicalStore::scalar() const
{
  LEGATE_ABORT("Not supported");
  return {};
}

template <typename T, std::int32_t DIM>
void PhysicalStore::bind_data(Buffer<T, DIM>&, const Point<DIM>&) const
{
  LEGATE_ABORT("Not supported");
}

template <typename T>
void PhysicalStore::check_accessor_type_() const
{
  if (constexpr auto in_type = type_code_of_v<T>; in_type != code()) {
    check_accessor_type_(in_type, sizeof(T));
  }
}

template <typename ACC, typename T, std::int32_t DIM>
ACC PhysicalStore::create_region_instance_accessor_(const Rect<DIM>& bounds,
                                                    bool validate_type) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);

  auto&& [instance, fid] = get_region_instance_();

  if (transformed()) {
    const auto transform = get_inverse_transform_();

    return dim_dispatch(transform.transform.m,
                        detail::store_detail::RegionInstanceTransAccessorFn<ACC, T, DIM>{},
                        instance,
                        fid,
                        transform,
                        bounds);
  }
  return {instance, fid, bounds, /* actual_field_size */ sizeof(T), validate_type};
}

template <typename ACC, typename T, std::int32_t DIM>
ACC PhysicalStore::create_region_instance_reduction_accessor_(const Rect<DIM>& bounds,
                                                              bool validate_type) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);

  auto&& [pr, fid] = get_region_instance_();

  if (transformed()) {
    auto transform = get_inverse_transform_();
    return dim_dispatch(transform.transform.m,
                        detail::store_detail::RegionInstanceTransReductionAccessorFn<ACC, T, DIM>{},
                        pr,
                        fid,
                        transform,
                        bounds);
  }
  return {pr,
          fid,
          bounds,
          /* silence_warnings */ false,
          /* warning_string */ nullptr,
          /* subfield_offset */ 0,
          /* actual_field_size */ sizeof(T),
          /* check_field_size */ validate_type};
}

inline const SharedPtr<detail::PhysicalStore>& PhysicalStore::impl() const { return impl_; }

}  // namespace legate
