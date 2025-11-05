/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_stores/unbound_physical_store.h>

#include <legate/data/detail/buffer.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/detail/traced_exception.h>

namespace legate::detail {

ReturnValue UnboundPhysicalStore::pack_weight() const { return unbound_field_.pack_weight(); }

Domain UnboundPhysicalStore::domain() const
{
  throw TracedException<std::invalid_argument>{
    "Invalid to retrieve the domain of an unbound store"};
}

InlineAllocation UnboundPhysicalStore::get_inline_allocation() const
{
  throw TracedException<std::invalid_argument>{
    "Allocation info cannot be retrieved from an unbound store"};
}

mapping::StoreTarget UnboundPhysicalStore::target() const
{
  if (unbound_field_.bound()) {
    return mapping::detail::to_target(unbound_field_.get_output_region().target_memory().kind());
  }
  throw TracedException<std::invalid_argument>{"Target of an unbound store cannot be queried"};
}

void UnboundPhysicalStore::bind_empty_data()
{
  check_valid_binding(true);
  unbound_field_.bind_empty_data(dim());
}

void UnboundPhysicalStore::bind_untyped_data(Buffer<std::int8_t, 1>& buffer,
                                             const Point<1>& extents)
{
  check_valid_binding(true);
  check_buffer_dimension(1);

  auto [out, fid] = get_output_field();

  out.return_data(DomainPoint{extents}, fid, buffer.get_instance(), false /*check_constraints*/);

  // We will use this value only when the unbound store is 1D
  update_num_elements(extents[0]);
}

void UnboundPhysicalStore::bind_data(const InternalSharedPtr<TaskLocalBuffer>& buffer,
                                     const DomainPoint& extents)
{
  check_valid_binding(/* bind_buffer */ true);
  check_buffer_dimension(extents.get_dim());

  auto [out, fid] = get_output_field();

  out.return_data(extents, fid, buffer->legion_buffer().get_instance());
  // We will use this value only when the unbound store is 1D
  update_num_elements(static_cast<std::size_t>(extents[0]));
}

void UnboundPhysicalStore::check_valid_binding(bool bind_buffer) const
{
  if (bind_buffer && unbound_field_.bound()) {
    throw TracedException<std::invalid_argument>{"A buffer has already been bound to the store"};
  }
}

void UnboundPhysicalStore::check_buffer_dimension(std::int32_t dim) const
{
  if (dim != this->dim()) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "Dimension mismatch: invalid to bind a {}-D buffer to a {}-D store", dim, this->dim())};
  }
}

void UnboundPhysicalStore::update_num_elements(std::size_t num_elements)
{
  unbound_field_.update_num_elements(num_elements);
  unbound_field_.set_bound(true);
}

}  // namespace legate::detail
