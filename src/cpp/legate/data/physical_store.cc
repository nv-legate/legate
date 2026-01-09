/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/physical_store.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/physical_stores/future_physical_store.h>
#include <legate/data/detail/physical_stores/region_physical_store.h>
#include <legate/data/detail/physical_stores/unbound_physical_store.h>
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
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "Data can only be bound to unbound stores"};
  }
  impl()->as_unbound_store().bind_data(buffer.impl(), extents);
}

void PhysicalStore::bind_untyped_data(Buffer<std::int8_t, 1>& buffer, const Point<1>& extents) const
{
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "Untyped data can only be bound to unbound stores"};
  }
  impl()->as_unbound_store().bind_untyped_data(buffer, extents);
}

void PhysicalStore::bind_empty_data() const
{
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "Empty data can only be bound to unbound stores"};
  }
  impl()->as_unbound_store().bind_empty_data();
}

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

bool PhysicalStore::is_future() const
{
  return impl()->kind() == detail::PhysicalStore::Kind::FUTURE;
}

bool PhysicalStore::is_unbound_store() const
{
  return impl()->kind() == detail::PhysicalStore::Kind::UNBOUND;
}

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
  impl()->check_accessor_type(code, size_of_T);
}

void PhysicalStore::check_accessor_dimension_(std::int32_t dim) const
{
  impl()->check_accessor_dimension(dim);
}

void PhysicalStore::check_accessor_store_backing_() const
{
  impl()->check_accessor_store_backing();
}

void PhysicalStore::check_shape_dimension_(std::int32_t dim) const
{
  impl()->check_shape_dimension(dim);
}

void PhysicalStore::check_valid_binding_(bool bind_buffer) const
{
  impl()->as_unbound_store().check_valid_binding(bind_buffer);
}

void PhysicalStore::check_buffer_dimension_(std::int32_t dim) const
{
  impl()->as_unbound_store().check_buffer_dimension(dim);
}

void PhysicalStore::check_write_access_() const { impl()->check_write_access(); }

void PhysicalStore::check_reduction_access_() const { impl()->check_reduction_access(); }

void PhysicalStore::check_scalar_store_() const { impl()->check_scalar_store(); }

void PhysicalStore::check_unbound_store_() const { impl()->check_unbound_store(); }

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform_() const
{
  return impl()->as_region_store().get_inverse_transform();
}

bool PhysicalStore::is_read_only_future_() const
{
  return impl()->as_future_store().is_read_only_future();
}

std::size_t PhysicalStore::get_field_offset_() const
{
  return impl()->as_future_store().get_field_offset();
}

const void* PhysicalStore::get_untyped_pointer_from_future_() const
{
  return impl()->as_future_store().get_untyped_pointer_from_future();
}

std::pair<Legion::PhysicalRegion, Legion::FieldID> PhysicalStore::get_region_field_() const
{
  return impl()->as_region_store().get_region_field();
}

GlobalRedopID PhysicalStore::get_redop_id_() const { return impl()->get_redop_id(); }

const Legion::Future& PhysicalStore::get_future_() const
{
  return impl()->as_future_store().get_future();
}

const Legion::UntypedDeferredValue& PhysicalStore::get_buffer_() const
{
  return impl()->as_future_store().get_buffer();
}

std::pair<Legion::OutputRegion, Legion::FieldID> PhysicalStore::get_output_field_() const
{
  return impl()->as_unbound_store().get_output_field();
}

void PhysicalStore::update_num_elements_(std::size_t num_elements) const
{
  impl()->as_unbound_store().update_num_elements(num_elements);
}

}  // namespace legate
