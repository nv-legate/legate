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

#include "core/mapping/array.h"

#include "core/mapping/detail/array.h"

namespace legate::mapping {

bool Array::nullable() const { return impl_->nullable(); }

int32_t Array::dim() const { return impl_->dim(); }

Type Array::type() const { return Type{impl_->type()}; }

Store Array::data() const { return Store{impl_->data().get()}; }

Store Array::null_mask() const { return Store{impl_->null_mask().get()}; }

std::vector<Store> Array::stores() const
{
  std::vector<Store> result;

  result.push_back(data());
  if (nullable()) {
    result.push_back(null_mask());
  }
  return result;
}

Domain Array::domain() const { return impl_->domain(); }

}  // namespace legate::mapping
