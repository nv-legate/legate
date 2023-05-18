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

#include "core/partitioning/restriction.h"

namespace legate {

Restriction join(Restriction lhs, Restriction rhs) { return std::max<Restriction>(lhs, rhs); }

tuple<Restriction> join(const tuple<Restriction>& lhs, const tuple<Restriction>& rhs)
{
  auto result = lhs;
  join_inplace(result, rhs);
  return std::move(result);
}

void join_inplace(Restrictions& lhs, const Restrictions& rhs)
{
  if (lhs.size() != rhs.size()) throw std::invalid_argument("Restrictions must have the same size");
  for (uint32_t idx = 0; idx < lhs.size(); ++idx) lhs[idx] = join(lhs[idx], rhs[idx]);
}

}  // namespace legate
