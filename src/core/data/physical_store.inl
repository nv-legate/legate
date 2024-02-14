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

// Useful for IDEs
#include "core/data/physical_store.h"

namespace legate::detail::store_detail {

template <typename ACC, typename T, std::int32_t N>
struct trans_accessor_fn {
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
                 std::int32_t redop_id,
                 const Legion::AffineTransform<M, N>& transform,
                 const Rect<N>& bounds)
  {
    return {pr, fid, redop_id, transform, bounds, false, nullptr, 0, sizeof(T), false};
  }
};

}  // namespace legate::detail::store_detail

namespace legate {

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRO<T, DIM> PhysicalStore::read_accessor() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) {
    if (is_read_only_future()) {
      return {get_future(), shape<DIM>(), Memory::Kind::NO_MEMKIND, sizeof(T), false};
    }
    return {get_buffer(), shape<DIM>(), sizeof(T), false};
  }

  return create_field_accessor<AccessorRO<T, DIM>, T, DIM>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorWO<T, DIM> PhysicalStore::write_accessor() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  check_write_access();

  if (is_future()) {
    return {get_buffer(), shape<DIM>(), sizeof(T), false};
  }

  return create_field_accessor<AccessorWO<T, DIM>, T, DIM>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRW<T, DIM> PhysicalStore::read_write_accessor() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  check_write_access();

  if (is_future()) {
    return {get_buffer(), shape<DIM>(), sizeof(T), false};
  }

  return create_field_accessor<AccessorRW<T, DIM>, T, DIM>(shape<DIM>());
}

template <typename OP, bool EXCLUSIVE, int DIM, bool VALIDATE_TYPE>
AccessorRD<OP, EXCLUSIVE, DIM> PhysicalStore::reduce_accessor() const
{
  using T = typename OP::LHS;
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  check_reduction_access();

  if (is_future()) {
    return {get_buffer(), shape<DIM>(), false, nullptr, 0, sizeof(T), false};
  }

  return create_reduction_accessor<AccessorRD<OP, EXCLUSIVE, DIM>, T, DIM>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRO<T, DIM> PhysicalStore::read_accessor(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) {
    if (is_read_only_future()) {
      return {get_future(), bounds, Memory::Kind::NO_MEMKIND, sizeof(T), false};
    }
    return {get_buffer(), bounds, sizeof(T), false};
  }

  return create_field_accessor<AccessorRO<T, DIM>, T, DIM>(bounds);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorWO<T, DIM> PhysicalStore::write_accessor(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  check_write_access();

  if (is_future()) {
    return {get_buffer(), bounds, sizeof(T), false};
  }

  return create_field_accessor<AccessorWO<T, DIM>, T, DIM>(bounds);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRW<T, DIM> PhysicalStore::read_write_accessor(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  check_write_access();

  if (is_future()) {
    return {get_buffer(), bounds, sizeof(T), false};
  }

  return create_field_accessor<AccessorRW<T, DIM>, T, DIM>(bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM, bool VALIDATE_TYPE>
AccessorRD<OP, EXCLUSIVE, DIM> PhysicalStore::reduce_accessor(const Rect<DIM>& bounds) const
{
  using T = typename OP::LHS;
  static_assert(DIM <= LEGATE_MAX_DIM);
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  check_reduction_access();

  if (is_future()) {
    return {get_buffer(), bounds, false, nullptr, 0, sizeof(T), false};
  }

  return create_reduction_accessor<AccessorRD<OP, EXCLUSIVE, DIM>, T, DIM>(bounds);
}

template <typename T, std::int32_t DIM>
Buffer<T, DIM> PhysicalStore::create_output_buffer(const Point<DIM>& extents,
                                                   bool bind_buffer /*= false*/) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  check_valid_binding(bind_buffer);
  check_buffer_dimension(DIM);

  Legion::OutputRegion out;
  Legion::FieldID fid;

  get_output_field(out, fid);

  auto result = out.create_buffer<T, DIM>(extents, fid, nullptr, bind_buffer);
  // We will use this value only when the unbound store is 1D
  if (bind_buffer) {
    update_num_elements(extents[0]);
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
  check_shape_dimension(DIM);
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
    throw std::invalid_argument("Scalars can be retrieved only from scalar stores");
  }
  if (is_read_only_future()) {
    return get_future().get_result<VAL>();
  }

  return get_buffer().operator Legion::DeferredValue<VAL>().read();
}

template <typename T, std::int32_t DIM>
void PhysicalStore::bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  check_valid_binding(true);
  check_buffer_dimension(DIM);

  Legion::OutputRegion out;
  Legion::FieldID fid;

  get_output_field(out, fid);

  out.return_data(extents, fid, buffer);
  // We will use this value only when the unbound store is 1D
  update_num_elements(extents[0]);
}

template <typename T>
void PhysicalStore::check_accessor_type() const
{
  auto in_type = type_code_of<T>;
  if (in_type == this->code()) {
    return;
  }
  // Test exact match for primitive types
  if (in_type != Type::Code::NIL) {
    throw std::invalid_argument{
      "Type mismatch: " + primitive_type(in_type).to_string() + " accessor to a " +
      type().to_string() +
      " store. Disable type checking via accessor template parameter if this is intended."};
  }
  // Test size matches for other types
  if (sizeof(T) != type().size()) {
    throw std::invalid_argument{
      "Type size mismatch: store type " + type().to_string() + " has size " +
      std::to_string(type().size()) + ", requested type has size " + std::to_string(sizeof(T)) +
      ". Disable type checking via accessor template parameter if this is intended."};
  }
}

template <typename ACC, typename T, std::int32_t DIM>
ACC PhysicalStore::create_field_accessor(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  Legion::PhysicalRegion pr;
  Legion::FieldID fid;

  get_region_field(pr, fid);
  if (transformed()) {
    auto transform = get_inverse_transform();
    return dim_dispatch(transform.transform.m,
                        detail::store_detail::trans_accessor_fn<ACC, T, DIM>{},
                        pr,
                        fid,
                        transform,
                        bounds);
  }
  return {pr, fid, bounds};
}

template <typename ACC, typename T, std::int32_t DIM>
ACC PhysicalStore::create_reduction_accessor(const Rect<DIM>& bounds) const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  Legion::PhysicalRegion pr;
  Legion::FieldID fid;

  get_region_field(pr, fid);
  if (transformed()) {
    auto transform = get_inverse_transform();
    return dim_dispatch(transform.transform.m,
                        detail::store_detail::trans_accessor_fn<ACC, T, DIM>{},
                        pr,
                        fid,
                        get_redop_id(),
                        transform,
                        bounds);
  }
  return {pr, fid, get_redop_id(), bounds};
}

inline PhysicalStore::PhysicalStore(InternalSharedPtr<detail::PhysicalStore> impl)
  : impl_{std::move(impl)}
{
}

inline const SharedPtr<detail::PhysicalStore>& PhysicalStore::impl() const { return impl_; }

}  // namespace legate
