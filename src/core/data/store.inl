/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

// Useful for IDEs
#include "core/data/store.h"

namespace legate::detail {
namespace {

template <typename ACC, int32_t N>
struct trans_accessor_fn {
  template <int32_t M>
  ACC operator()(const Legion::PhysicalRegion& pr,
                 Legion::FieldID fid,
                 const Legion::AffineTransform<M, N>& transform,
                 const Rect<N>& bounds)
  {
    return ACC(pr, fid, transform, bounds);
  }
  template <int32_t M>
  ACC operator()(const Legion::PhysicalRegion& pr,
                 Legion::FieldID fid,
                 int32_t redop_id,
                 const Legion::AffineTransform<M, N>& transform,
                 const Rect<N>& bounds)
  {
    return ACC(pr, fid, redop_id, transform, bounds);
  }
};

}  // namespace
}  // namespace legate::detail

namespace legate {

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRO<T, DIM> Store::read_accessor() const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) {
    if (is_read_only_future()) {
      return AccessorRO<T, DIM>(get_future(), shape<DIM>(), Memory::Kind::NO_MEMKIND);
    } else {
      return AccessorRO<T, DIM>(get_buffer(), shape<DIM>());
    }
  }

  return create_field_accessor<AccessorRO<T, DIM>, DIM>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorWO<T, DIM> Store::write_accessor() const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) { return AccessorWO<T, DIM>(get_buffer(), shape<DIM>()); }

  return create_field_accessor<AccessorWO<T, DIM>, DIM>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRW<T, DIM> Store::read_write_accessor() const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) { return AccessorRW<T, DIM>(get_buffer(), shape<DIM>()); }

  return create_field_accessor<AccessorRW<T, DIM>, DIM>(shape<DIM>());
}

template <typename OP, bool EXCLUSIVE, int DIM, bool VALIDATE_TYPE>
AccessorRD<OP, EXCLUSIVE, DIM> Store::reduce_accessor() const
{
  using T = typename OP::LHS;
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) { return AccessorRD<OP, EXCLUSIVE, DIM>(get_buffer(), shape<DIM>()); }

  return create_reduction_accessor<AccessorRD<OP, EXCLUSIVE, DIM>, DIM>(shape<DIM>());
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRO<T, DIM> Store::read_accessor(const Rect<DIM>& bounds) const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) {
    if (is_read_only_future()) {
      return AccessorRO<T, DIM>(get_future(), bounds, Memory::Kind::NO_MEMKIND);
    } else {
      return AccessorRO<T, DIM>(get_buffer(), bounds);
    }
  }

  return create_field_accessor<AccessorRO<T, DIM>, DIM>(bounds);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorWO<T, DIM> Store::write_accessor(const Rect<DIM>& bounds) const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) { return AccessorWO<T, DIM>(get_buffer(), bounds); }

  return create_field_accessor<AccessorWO<T, DIM>, DIM>(bounds);
}

template <typename T, int DIM, bool VALIDATE_TYPE>
AccessorRW<T, DIM> Store::read_write_accessor(const Rect<DIM>& bounds) const
{
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) { return AccessorRW<T, DIM>(get_buffer(), bounds); }

  return create_field_accessor<AccessorRW<T, DIM>, DIM>(bounds);
}

template <typename OP, bool EXCLUSIVE, int DIM, bool VALIDATE_TYPE>
AccessorRD<OP, EXCLUSIVE, DIM> Store::reduce_accessor(const Rect<DIM>& bounds) const
{
  using T = typename OP::LHS;
  if constexpr (VALIDATE_TYPE) {
    check_accessor_dimension(DIM);
    check_accessor_type<T>();
  }

  if (is_future()) { return AccessorRD<OP, EXCLUSIVE, DIM>(get_buffer(), bounds); }

  return create_reduction_accessor<AccessorRD<OP, EXCLUSIVE, DIM>, DIM>(bounds);
}

template <typename T, int32_t DIM>
Buffer<T, DIM> Store::create_output_buffer(const Point<DIM>& extents, bool bind_buffer /*= false*/)
{
  check_valid_binding(bind_buffer);
  check_buffer_dimension(DIM);

  Legion::OutputRegion out;
  Legion::FieldID fid;
  get_output_field(out, fid);

  auto result = out.create_buffer<T, DIM>(extents, fid, nullptr, bind_buffer);
  // We will use this value only when the unbound store is 1D
  if (bind_buffer) update_num_elements(extents[0]);
  return result;
}

template <int32_t DIM>
Rect<DIM> Store::shape() const
{
  check_shape_dimension(DIM);
  if (dim() > 0) {
    return domain().bounds<DIM, Legion::coord_t>();
  } else {
    auto p = Point<DIM>::ZEROES();
    return Rect<DIM>(p, p);
  }
}

template <typename VAL>
VAL Store::scalar() const
{
  if (!is_future()) {
    throw std::invalid_argument("Scalars can be retrieved only from scalar stores");
  }
  if (is_read_only_future()) {
    return get_future().get_result<VAL>();
  } else {
    return get_buffer().operator Legion::DeferredValue<VAL>().read();
  }
}

template <typename T, int32_t DIM>
void Store::bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents)
{
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
void Store::check_accessor_type() const
{
  auto in_type = legate_type_code_of<T>;
  if (in_type == this->code()) return;
  // Test exact match for primitive types
  if (in_type != Type::Code::INVALID) {
    throw std::invalid_argument(
      "Type mismatch: " + primitive_type(in_type).to_string() + " accessor to a " +
      type().to_string() +
      " store. Disable type checking via accessor template parameter if this is intended.");
  }
  // Test size matches for other types
  if (sizeof(T) != type().size()) {
    throw std::invalid_argument(
      "Type size mismatch: store type " + type().to_string() + " has size " +
      std::to_string(type().size()) + ", requested type has size " + std::to_string(sizeof(T)) +
      ". Disable type checking via accessor template parameter if this is intended.");
  }
}

template <typename ACC, int32_t DIM>
ACC Store::create_field_accessor(const Rect<DIM>& bounds) const
{
  Legion::PhysicalRegion pr;
  Legion::FieldID fid;
  get_region_field(pr, fid);

  if (transformed()) {
    auto transform = get_inverse_transform();
    return dim_dispatch(
      transform.transform.m, detail::trans_accessor_fn<ACC, DIM>{}, pr, fid, transform, bounds);
  }
  return ACC(pr, fid, bounds);
}

template <typename ACC, int32_t DIM>
ACC Store::create_reduction_accessor(const Rect<DIM>& bounds) const
{
  Legion::PhysicalRegion pr;
  Legion::FieldID fid;
  get_region_field(pr, fid);

  if (transformed()) {
    auto transform = get_inverse_transform();
    return dim_dispatch(transform.transform.m,
                        detail::trans_accessor_fn<ACC, DIM>{},
                        pr,
                        fid,
                        get_redop_id(),
                        transform,
                        bounds);
  }
  return ACC(pr, fid, get_redop_id(), bounds);
}

}  // namespace legate
