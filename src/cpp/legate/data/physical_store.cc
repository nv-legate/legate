/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/physical_store.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/logical_store.h>
#include <legate/data/physical_array.h>
#include <legate/utilities/detail/dlpack/to_dlpack.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate {

PhysicalStore::PhysicalStore(InternalSharedPtr<detail::PhysicalStore> impl,
                             std::optional<LogicalStore> owner)
  : impl_{std::move(impl)}, owner_{std::move(owner)}
{
}

/*static*/ void PhysicalStore::throw_invalid_scalar_access_()
{
  throw detail::TracedException<std::invalid_argument>{
    "Scalars can be retrieved only from scalar stores"};
}

namespace {

class CreateBuffer {
 public:
  template <Type::Code CODE, std::int32_t DIM>
  [[nodiscard]] TaskLocalBuffer operator()(const PhysicalStore& store,
                                           const Type& ty,
                                           const DomainPoint& extents,
                                           bool bind_buffer) const
  {
    const auto buf =
      store.create_output_buffer<type_of_t<CODE>>(static_cast<Point<DIM>>(extents), bind_buffer);

    return TaskLocalBuffer{buf, ty};
  }
};

}  // namespace

TaskLocalBuffer PhysicalStore::create_output_buffer(const DomainPoint& extents,
                                                    bool bind_buffer) const
{
  auto&& ty = type();

  return double_dispatch(
    extents.get_dim(), ty.code(), CreateBuffer{}, *this, ty, extents, bind_buffer);
}

void PhysicalStore::bind_data(const TaskLocalBuffer& buffer,
                              const DomainPoint& extents,
                              bool check_type) const
{
  if (check_type && (buffer.type() != type())) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Cannot bind data of type {} to store of type {}, types are not compatible.",
                  buffer.type(),
                  type())};
  }
  impl()->bind_data(buffer.impl(), extents);
}

void PhysicalStore::bind_untyped_data(Buffer<std::int8_t, 1>& buffer, const Point<1>& extents) const
{
  check_valid_binding_(true);
  check_buffer_dimension_(1);

  auto [out, fid] = get_output_field_();

  out.return_data(DomainPoint{extents}, fid, buffer.get_instance(), false /*check_constraints*/);

  // We will use this value only when the unbound store is 1D
  update_num_elements_(extents[0]);
}

void PhysicalStore::bind_empty_data() const { impl()->bind_empty_data(); }

std::int32_t PhysicalStore::dim() const { return impl()->dim(); }

Type PhysicalStore::type() const { return Type{impl()->type()}; }

Domain PhysicalStore::domain() const { return impl()->domain(); }

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  return impl()->get_inline_allocation();
}

bool PhysicalStore::is_readable() const { return impl()->is_readable(); }

bool PhysicalStore::is_writable() const { return impl()->is_writable(); }

bool PhysicalStore::is_reducible() const { return impl()->is_reducible(); }

bool PhysicalStore::valid() const { return impl() != nullptr && impl()->valid(); }

bool PhysicalStore::transformed() const { return impl()->transformed(); }

bool PhysicalStore::is_future() const { return impl()->is_future(); }

bool PhysicalStore::is_unbound_store() const { return impl()->is_unbound_store(); }

bool PhysicalStore::is_partitioned() const { return impl()->is_partitioned(); }

mapping::StoreTarget PhysicalStore::target() const { return impl()->target(); }

std::unique_ptr<DLManagedTensorVersioned, void (*)(DLManagedTensorVersioned*)>
PhysicalStore::to_dlpack(std::optional<bool> copy, std::optional<CUstream_st*> stream) const
{
  return detail::to_dlpack(*this, std::move(copy), std::move(stream));
}

PhysicalStore::PhysicalStore(const PhysicalArray& array)
  : PhysicalStore{
      array.nullable() ? throw detail::TracedException<
                           std::invalid_argument>{"Nullable array cannot be converted to a store"}
                       : array.data().impl(),
      array.owner().has_value() ? std::make_optional(array.owner()->data()) : std::nullopt}
{
}

void PhysicalStore::check_accessor_type_(Type::Code code, std::size_t size_of_T) const
{
  // Test exact match for primitive types
  if (code != Type::Code::NIL) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Type mismatch: {} accessor to a {} store. Disable type checking via accessor "
                  "template parameter if this is intended.",
                  primitive_type(code).to_string(),
                  type().to_string())};
  }
  // Test size matches for other types
  if (size_of_T != type().size()) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Type size mismatch: store type {} has size {}, requested type has size {}. "
                  "Disable type checking via accessor template parameter if this is intended.",
                  type().to_string(),
                  type().size(),
                  size_of_T)};
  }
}

void PhysicalStore::check_accessor_dimension_(std::int32_t dim) const
{
  impl()->check_accessor_dimension_(dim);
}

void PhysicalStore::check_buffer_dimension_(std::int32_t dim) const
{
  impl()->check_buffer_dimension_(dim);
}

void PhysicalStore::check_shape_dimension_(std::int32_t dim) const
{
  impl()->check_shape_dimension_(dim);
}

void PhysicalStore::check_valid_binding_(bool bind_buffer) const
{
  impl()->check_valid_binding_(bind_buffer);
}

void PhysicalStore::check_write_access_() const { impl()->check_write_access_(); }

void PhysicalStore::check_reduction_access_() const { impl()->check_reduction_access_(); }

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform_() const
{
  return impl()->get_inverse_transform_();
}

bool PhysicalStore::is_read_only_future_() const { return impl()->is_read_only_future_(); }

std::size_t PhysicalStore::get_field_offset_() const { return impl()->get_field_offset_(); }

const void* PhysicalStore::get_untyped_pointer_from_future_() const
{
  return impl()->get_untyped_pointer_from_future_();
}

std::pair<Legion::PhysicalRegion, Legion::FieldID> PhysicalStore::get_region_field_() const
{
  return impl()->get_region_field_();
}

GlobalRedopID PhysicalStore::get_redop_id_() const { return impl()->get_redop_id_(); }

const Legion::Future& PhysicalStore::get_future_() const { return impl()->get_future(); }

const Legion::UntypedDeferredValue& PhysicalStore::get_buffer_() const
{
  return impl()->get_buffer();
}

std::pair<Legion::OutputRegion, Legion::FieldID> PhysicalStore::get_output_field_() const
{
  return impl()->get_output_field_();
}

void PhysicalStore::update_num_elements_(std::size_t num_elements) const
{
  impl()->update_num_elements_(num_elements);
}

}  // namespace legate
