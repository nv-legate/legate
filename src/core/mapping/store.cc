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

#include "core/mapping/store.h"

#include "core/mapping/detail/store.h"

namespace legate::mapping {

bool Store::is_future() const { return impl()->is_future(); }

bool Store::unbound() const { return impl()->unbound(); }

std::uint32_t Store::dim() const { return impl()->dim(); }

bool Store::is_reduction() const { return impl()->is_reduction(); }

std::int32_t Store::redop() const { return impl()->redop(); }

bool Store::can_colocate_with(const Store& other) const
{
  return impl()->can_colocate_with(*other.impl());
}

Domain Store::domain() const { return impl()->domain(); }

}  // namespace legate::mapping
