/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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
