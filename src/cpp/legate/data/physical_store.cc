/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/physical_store.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/logical_store.h>
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

TaskLocalBuffer PhysicalStore::create_output_buffer(const DomainPoint&, bool) const
{
  return TaskLocalBuffer{nullptr};
}

void PhysicalStore::bind_data(const TaskLocalBuffer& buffer,
                              const DomainPoint&,
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
}

void PhysicalStore::bind_untyped_data(Buffer<std::int8_t, 1>&, const Point<1>&) const
{
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "Untyped data can only be bound to unbound stores"};
  }
}

void PhysicalStore::bind_empty_data() const
{
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "Empty data can only be bound to unbound stores"};
  }
}

std::int32_t PhysicalStore::dim() const { return impl()->dim(); }

Type PhysicalStore::type() const { return Type{impl()->type()}; }

Domain PhysicalStore::domain() const { return impl()->domain(); }

LogicalStore PhysicalStore::to_logical_store() const
{
  return LogicalStore{impl_->to_logical_store(impl_)};
}

InlineAllocation PhysicalStore::get_inline_allocation() const
{
  return impl()->get_inline_allocation();
}

bool PhysicalStore::is_readable() const { return impl()->is_readable(); }

bool PhysicalStore::is_writable() const { return impl()->is_writable(); }

bool PhysicalStore::is_reducible() const { return impl()->is_reducible(); }

bool PhysicalStore::valid() const { return impl() != nullptr && impl()->valid(); }

bool PhysicalStore::transformed() const { return impl()->transformed(); }

bool PhysicalStore::is_future() const { return false; }

bool PhysicalStore::is_unbound_store() const { return false; }

mapping::StoreTarget PhysicalStore::target() const { return impl()->target(); }

std::unique_ptr<DLManagedTensorVersioned, void (*)(DLManagedTensorVersioned*)>
PhysicalStore::to_dlpack(std::optional<bool> copy, std::optional<CUstream_st*> stream) const
{
  return detail::to_dlpack(*this, std::move(copy), std::move(stream));
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

void PhysicalStore::check_write_access_() const { impl()->check_write_access(); }

void PhysicalStore::check_reduction_access_() const { impl()->check_reduction_access(); }

Legion::DomainAffineTransform PhysicalStore::get_inverse_transform_() const
{
  return impl()->get_inverse_transform();
}

GlobalRedopID PhysicalStore::get_redop_id_() const { return impl()->get_redop_id(); }

std::pair<Realm::RegionInstance, Realm::FieldID> PhysicalStore::get_region_instance_() const
{
  return impl()->get_region_instance();
}

}  // namespace legate
