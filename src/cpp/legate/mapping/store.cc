/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/store.h>

#include <legate/mapping/detail/store.h>

namespace legate::mapping {

bool Store::is_future() const { return impl()->is_future(); }

bool Store::unbound() const { return impl()->unbound(); }

std::uint32_t Store::dim() const { return impl()->dim(); }

bool Store::is_reduction() const { return impl()->is_reduction(); }

GlobalRedopID Store::redop() const { return impl()->redop(); }

bool Store::can_colocate_with(const Store& other) const
{
  return impl()->can_colocate_with(*other.impl());
}

Domain Store::domain() const { return impl()->domain(); }

}  // namespace legate::mapping
