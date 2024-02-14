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

#include "core/data/physical_array.h"

#include "core/data/detail/array_kind.h"
#include "core/data/detail/physical_array.h"
#include "core/data/physical_store.h"
#include "core/type/type_info.h"
#include "core/utilities/typedefs.h"

#include <cstdint>
#include <stdexcept>

namespace legate {

bool PhysicalArray::nullable() const noexcept { return impl_->nullable(); }

std::int32_t PhysicalArray::dim() const noexcept { return impl_->dim(); }

Type PhysicalArray::type() const { return Type{impl_->type()}; }

bool PhysicalArray::nested() const noexcept { return impl_->nested(); }

PhysicalStore PhysicalArray::data() const { return PhysicalStore{impl_->data()}; }

PhysicalStore PhysicalArray::null_mask() const { return PhysicalStore{impl_->null_mask()}; }

PhysicalArray PhysicalArray::child(std::uint32_t index) const
{
  return PhysicalArray{impl_->child(index)};
}

Domain PhysicalArray::domain() const { return impl_->domain(); }

void PhysicalArray::check_shape_dimension(std::int32_t dim) const
{
  impl_->check_shape_dimension(dim);
}

ListPhysicalArray PhysicalArray::as_list_array() const
{
  if (impl_->kind() != detail::ArrayKind::LIST) {
    throw std::invalid_argument{"Array is not a list array"};
  }
  return ListPhysicalArray{impl_};
}

StringPhysicalArray PhysicalArray::as_string_array() const
{
  if (type().code() != Type::Code::STRING) {
    throw std::invalid_argument{"Array is not a string array"};
  }
  return StringPhysicalArray{impl_};
}

PhysicalArray ListPhysicalArray::descriptor() const
{
  return PhysicalArray{static_cast<const detail::ListPhysicalArray*>(impl_.get())->descriptor()};
}

PhysicalArray ListPhysicalArray::vardata() const
{
  return PhysicalArray{static_cast<const detail::ListPhysicalArray*>(impl_.get())->vardata()};
}

PhysicalArray StringPhysicalArray::ranges() const
{
  return PhysicalArray{static_cast<const detail::ListPhysicalArray*>(impl_.get())->descriptor()};
}

PhysicalArray StringPhysicalArray::chars() const
{
  return PhysicalArray{static_cast<const detail::ListPhysicalArray*>(impl_.get())->vardata()};
}

}  // namespace legate
