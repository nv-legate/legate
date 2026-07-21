/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/physical_store.h>

#include <legate/data/detail/buffer.h>
#include <legate/data/detail/inline_storage.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/logical_store.h>
#include <legate/utilities/detail/dlpack/to_dlpack.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/dispatch.h>

#include <fmt/format.h>

#include <cstdint>
#include <stdexcept>
#include <utility>

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

namespace {

// Domain spanning `[0, extents)` in every dimension.
[[nodiscard]] Domain extents_to_domain(const DomainPoint& extents)
{
  auto lo = DomainPoint::zeroes(static_cast<std::int32_t>(extents.dim()));
  auto hi = DomainPoint::zeroes(static_cast<std::int32_t>(extents.dim()));

  for (std::uint32_t i = 0; i < extents.dim(); ++i) {
    lo[i] = 0;
    hi[i] = extents[i] - 1;
  }
  return Domain{std::move(lo), std::move(hi)};
}

}  // namespace

TaskLocalBuffer PhysicalStore::create_output_buffer(const DomainPoint& extents,
                                                    bool bind_buffer) const
{
  return double_dispatch(static_cast<std::int32_t>(extents.dim()),
                         type().code(),
                         CreateBuffer{},
                         *this,
                         type(),
                         extents,
                         bind_buffer);
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
  bind_inline_storage_(buffer.impl()->get_storage(), extents_to_domain(extents));
}

void PhysicalStore::bind_untyped_data(Buffer<std::int8_t, 1>&, const Point<1>&) const
{
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "Untyped data can only be bound to unbound stores"};
  }
  // Binding a byte buffer to a store of a wider element type requires reinterpreting the buffer's
  // allocation with the store's field size, which the single-rank POC does not implement yet. Use
  // the typed bind_data()/create_output_buffer() paths instead.
  throw detail::TracedException<std::runtime_error>{
    "bind_untyped_data is not supported in this single-rank POC; use the typed bind_data()"};
}

void PhysicalStore::bind_empty_data() const
{
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "Empty data can only be bound to unbound stores"};
  }
  impl()->bind_empty_data();
}

InternalSharedPtr<detail::InlineStorage> PhysicalStore::create_output_buffer_storage_(
  const Domain& bounds) const
{
  return impl()->create_output_buffer_storage(bounds);
}

void PhysicalStore::bind_inline_storage_(const InternalSharedPtr<detail::InlineStorage>& storage,
                                         const Domain& extents) const
{
  if (!is_unbound_store()) {
    throw detail::TracedException<std::invalid_argument>{
      "Data can only be bound to unbound stores"};
  }
  impl()->bind_data(storage, extents);
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

bool PhysicalStore::is_unbound_store() const { return impl()->is_unbound_store(); }

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

void PhysicalStore::check_shape_dimension_(std::int32_t dim) const
{
  impl()->check_shape_dimension(dim);
}

void PhysicalStore::check_write_access_() const { impl()->check_write_access(); }

void PhysicalStore::check_reduction_access_() const { impl()->check_reduction_access(); }

void* PhysicalStore::base_() const { return impl()->base(); }

const DomainPoint& PhysicalStore::strides_() const { return impl()->strides(); }

}  // namespace legate
