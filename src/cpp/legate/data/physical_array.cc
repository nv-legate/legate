/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/physical_array.h>

#include <legate/data/detail/physical_array.h>
#include <legate/data/detail/physical_arrays/list_physical_array.h>
#include <legate/data/logical_array.h>
#include <legate/data/physical_store.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <stdexcept>

namespace legate {

PhysicalArray::PhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl,
                             std::optional<LogicalArray> owner)
  : impl_{std::move(impl)}, owner_{std::move(owner)}
{
}

bool PhysicalArray::nullable() const noexcept { return impl()->nullable(); }

std::int32_t PhysicalArray::dim() const noexcept { return impl()->dim(); }

Type PhysicalArray::type() const noexcept { return Type{impl()->type()}; }

bool PhysicalArray::nested() const noexcept { return impl()->nested(); }

PhysicalStore PhysicalArray::data() const
{
  return PhysicalStore{impl()->data(),
                       owner().has_value() ? std::make_optional(owner()->data()) : std::nullopt};
}

PhysicalStore PhysicalArray::null_mask() const
{
  return PhysicalStore{
    impl()->null_mask(),
    owner().has_value() ? std::make_optional(owner()->null_mask()) : std::nullopt};
}

PhysicalArray PhysicalArray::child(std::uint32_t index) const
{
  return PhysicalArray{
    impl()->child(index),
    owner().has_value() ? std::make_optional(owner()->child(index)) : std::nullopt};
}

Domain PhysicalArray::domain() const { return impl()->domain(); }

void PhysicalArray::check_shape_dimension_(std::int32_t dim) const
{
  impl()->check_shape_dimension(dim);
}

ListPhysicalArray PhysicalArray::as_list_array() const
{
  if (const auto* list_array = dynamic_cast<const detail::ListPhysicalArray*>(impl().get());
      !list_array) {
    throw detail::TracedException<std::invalid_argument>{"Array is not a list array"};
  }
  return ListPhysicalArray{impl(), owner()};
}

StringPhysicalArray PhysicalArray::as_string_array() const
{
  if (type().code() != Type::Code::STRING) {
    throw detail::TracedException<std::invalid_argument>{"Array is not a string array"};
  }
  return StringPhysicalArray{impl(), owner()};
}

// ==========================================================================================

ListPhysicalArray::ListPhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl,
                                     std::optional<LogicalArray> owner)
  : PhysicalArray{std::move(impl), std::move(owner)}
{
}

// ==========================================================================================

PhysicalArray ListPhysicalArray::descriptor() const
{
  return PhysicalArray{
    static_cast<const detail::ListPhysicalArray*>(impl().get())->descriptor(),
    owner().has_value() ? std::make_optional(owner()->as_list_array().descriptor()) : std::nullopt};
}

PhysicalArray ListPhysicalArray::vardata() const
{
  return PhysicalArray{
    static_cast<const detail::ListPhysicalArray*>(impl().get())->vardata(),
    owner().has_value() ? std::make_optional(owner()->as_list_array().vardata()) : std::nullopt};
}

// ==========================================================================================

StringPhysicalArray::StringPhysicalArray(InternalSharedPtr<detail::PhysicalArray> impl,
                                         std::optional<LogicalArray> owner)
  : PhysicalArray{std::move(impl), std::move(owner)}
{
}

PhysicalArray StringPhysicalArray::ranges() const
{
  return PhysicalArray{
    static_cast<const detail::ListPhysicalArray*>(impl().get())->descriptor(),
    owner().has_value() ? std::make_optional(owner()->as_string_array().offsets()) : std::nullopt};
}

PhysicalArray StringPhysicalArray::chars() const
{
  return PhysicalArray{
    static_cast<const detail::ListPhysicalArray*>(impl().get())->vardata(),
    owner().has_value() ? std::make_optional(owner()->as_string_array().chars()) : std::nullopt};
}

}  // namespace legate
