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

bool Store::is_future() const { return impl_->is_future(); }

bool Store::unbound() const { return impl_->unbound(); }

int32_t Store::dim() const { return impl_->dim(); }

bool Store::is_reduction() const { return impl_->is_reduction(); }

int32_t Store::redop() const { return impl_->redop(); }

bool Store::can_colocate_with(const Store& other) const
{
  return impl_->can_colocate_with(*other.impl_);
}

Domain Store::domain() const { return impl_->domain(); }

Store::Store(const detail::Store* impl) : impl_(impl) {}

Store::Store(const Store& other) = default;

Store& Store::operator=(const Store& other) = default;

Store::Store(Store&& other) = default;

Store& Store::operator=(Store&& other) = default;

// Don't need to release impl as it's held by the operation
Store::~Store() {}

}  // namespace legate::mapping
