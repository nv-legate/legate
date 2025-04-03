/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate/data/physical_store.h>

namespace legate::detail::store_detail {

template <typename ACC, typename T, std::int32_t N>
class TransAccessorFn {
 public:
  template <std::int32_t M>
  ACC operator()(const Legion::PhysicalRegion& pr,
                 Legion::FieldID fid,
                 const Legion::AffineTransform<M, N>& transform,
                 const Rect<N>& bounds)
  {
    return {pr, fid, transform, bounds, sizeof(T), false};
  }

  template <std::int32_t M>
  ACC operator()(const Legion::PhysicalRegion& pr,
                 Legion::FieldID fid,
                 GlobalRedopID redop_id,
                 const Legion::AffineTransform<M, N>& transform,
                 const Rect<N>& bounds)
  {
    return {pr,
            fid,
            static_cast<Legion::ReductionOpID>(redop_id),
            transform,
            bounds,
            false,
            nullptr,
            0,
            sizeof(T),
            false};
  }
};

}  // namespace legate::detail::store_detail

namespace legate {

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

  if (is_future()) {
    if (is_read_only_future_()) {
      return {get_future_(),
              bounds,
              Memory::Kind::NO_MEMKIND,
              sizeof(T),
              false,
              false,
              nullptr,
              get_field_offset_()};
    }
    return {get_buffer_(), bounds, sizeof(T), false};
  }

  return create_field_accessor_<AccessorRO<T, DIM>, T, DIM>(bounds);
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

  if (is_future()) {
    return {get_buffer_(), bounds, sizeof(T), false};
  }

  return create_field_accessor_<AccessorWO<T, DIM>, T, DIM>(bounds);
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

  if (is_future()) {
    return {get_buffer_(), bounds, sizeof(T), false};
  }

  return create_field_accessor_<AccessorRW<T, DIM>, T, DIM>(bounds);
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

  if (is_future()) {
    return {get_buffer_(), bounds, false, nullptr, 0, sizeof(T), false};
  }

  return create_reduction_accessor_<AccessorRD<OP, EXCLUSIVE, DIM>, T, DIM>(bounds);
}

template <typename T, std::int32_t DIM>
Buffer<T, DIM> PhysicalStore::create_output_buffer(const Point<DIM>& extents,
                                                   bool bind_buffer /*= false*/) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  check_valid_binding_(bind_buffer);
  check_buffer_dimension_(DIM);

  auto [out, fid] = get_output_field_();

  auto result = out.create_buffer<T, DIM>(extents, fid, nullptr, bind_buffer);
  // We will use this value only when the unbound store is 1D
  if (bind_buffer) {
    update_num_elements_(extents[0]);
  }
  return result;
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
  if (!is_future()) {
    throw_invalid_scalar_access_();
  }
  if (is_read_only_future_()) {
    // get_untyped_pointer_from_future_ is guaranteed to return an aligned pointer when T is the
    // right value type
    return *static_cast<const VAL*>(get_untyped_pointer_from_future_());
  }

  return get_buffer_().operator Legion::DeferredValue<VAL>().read();
}

template <typename T, std::int32_t DIM>
void PhysicalStore::bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  check_valid_binding_(true);
  check_buffer_dimension_(DIM);

  auto [out, fid] = get_output_field_();

  out.return_data(extents, fid, buffer);
  // We will use this value only when the unbound store is 1D
  update_num_elements_(extents[0]);
}

template <typename T>
void PhysicalStore::check_accessor_type_() const
{
  if (constexpr auto in_type = type_code_of_v<T>; in_type != code()) {
    check_accessor_type_(in_type, sizeof(T));
  }
}

template <typename ACC, typename T, std::int32_t DIM>
ACC PhysicalStore::create_field_accessor_(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);

  auto [pr, fid] = get_region_field_();

  if (transformed()) {
    auto transform = get_inverse_transform_();
    return dim_dispatch(transform.transform.m,
                        detail::store_detail::TransAccessorFn<ACC, T, DIM>{},
                        std::move(pr),
                        fid,
                        transform,
                        bounds);
  }
  return {std::move(pr), fid, bounds};
}

template <typename ACC, typename T, std::int32_t DIM>
ACC PhysicalStore::create_reduction_accessor_(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);

  auto [pr, fid] = get_region_field_();

  if (transformed()) {
    auto transform = get_inverse_transform_();
    return dim_dispatch(transform.transform.m,
                        detail::store_detail::TransAccessorFn<ACC, T, DIM>{},
                        std::move(pr),
                        fid,
                        get_redop_id_(),
                        transform,
                        bounds);
  }
  return {std::move(pr), fid, static_cast<Legion::ReductionOpID>(get_redop_id_()), bounds};
}

inline const SharedPtr<detail::PhysicalStore>& PhysicalStore::impl() const { return impl_; }

}  // namespace legate
