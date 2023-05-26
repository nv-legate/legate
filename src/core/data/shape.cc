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

#include "core/data/shape.h"

namespace legate {

Legion::Domain to_domain(const tuple<size_t>& shape)
{
  Legion::Domain domain;
  auto ndim  = static_cast<int32_t>(shape.size());
  domain.dim = ndim;
  for (int32_t idx = 0; idx < ndim; ++idx) {
    domain.rect_data[idx]        = 0;
    domain.rect_data[idx + ndim] = static_cast<int64_t>(shape[idx]) - 1;
  }
  return domain;
}

Legion::DomainPoint to_domain_point(const Shape& shape)
{
  Legion::DomainPoint point;
  auto ndim = static_cast<int32_t>(shape.size());
  point.dim = ndim;
  for (int32_t idx = 0; idx < ndim; ++idx) point[idx] = static_cast<int64_t>(shape[idx]);
  return point;
}

}  // namespace legate
