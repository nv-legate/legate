/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/array.h>

#include <legate/mapping/detail/array.h>

namespace legate::mapping {

bool Array::nullable() const { return impl_->nullable(); }

std::int32_t Array::dim() const { return impl_->dim(); }

Type Array::type() const { return Type{impl_->type()}; }

Store Array::data() const { return Store{impl_->data().get()}; }

Store Array::null_mask() const { return Store{impl_->null_mask().get()}; }

std::vector<Store> Array::stores() const
{
  auto raw_stores = impl_->stores();
  std::vector<Store> stores;
  stores.reserve(raw_stores.size());
  std::transform(
    raw_stores.begin(), raw_stores.end(), std::back_inserter(stores), [](const auto& x) {
      return Store{x.get()};
    });
  return stores;
}

Domain Array::domain() const { return impl_->domain(); }

}  // namespace legate::mapping
