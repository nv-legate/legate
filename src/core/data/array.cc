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

#include "core/data/array.h"

#include "core/data/detail/array.h"
#include "core/data/detail/array_kind.h"
#include "core/data/store.h"
#include "core/type/type_info.h"
#include "core/utilities/typedefs.h"

#include <cstdint>
#include <stdexcept>

namespace legate {

bool Array::nullable() const noexcept { return impl_->nullable(); }

int32_t Array::dim() const noexcept { return impl_->dim(); }

Type Array::type() const { return Type{impl_->type()}; }

bool Array::nested() const noexcept { return impl_->nested(); }

Store Array::data() const { return Store{impl_->data()}; }

Store Array::null_mask() const { return Store{impl_->null_mask()}; }

Array Array::child(uint32_t index) const { return Array{impl_->child(index)}; }

Domain Array::domain() const { return impl_->domain(); }

void Array::check_shape_dimension(int32_t dim) const { impl_->check_shape_dimension(dim); }

ListArray Array::as_list_array() const
{
  if (impl_->kind() != detail::ArrayKind::LIST) {
    throw std::invalid_argument{"Array is not a list array"};
  }
  return ListArray{impl_};
}

StringArray Array::as_string_array() const
{
  if (type().code() != Type::Code::STRING) {
    throw std::invalid_argument{"Array is not a string array"};
  }
  return StringArray{impl_};
}

Array ListArray::descriptor() const
{
  return Array{static_cast<const detail::ListArray*>(impl_.get())->descriptor()};
}

Array ListArray::vardata() const
{
  return Array{static_cast<const detail::ListArray*>(impl_.get())->vardata()};
}

Array StringArray::ranges() const
{
  return Array{static_cast<const detail::ListArray*>(impl_.get())->descriptor()};
}

Array StringArray::chars() const
{
  return Array{static_cast<const detail::ListArray*>(impl_.get())->vardata()};
}

}  // namespace legate
