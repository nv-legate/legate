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

#pragma once

#include "core/utilities/tuple.h"

namespace legate {

/**
 * @brief Enum to describe partitioning preference on dimensions of a store
 */
enum class Restriction : int32_t {
  ALLOW  = 0, /*!< The dimension can be partitioned */
  AVOID  = 1, /*!< The dimension can be partitioned, but other dimensions are preferred */
  FORBID = 2, /*!< The dimension must not be partitioned */
};

using Restrictions = tuple<Restriction>;

Restriction join(Restriction lhs, Restriction rhs);

tuple<Restriction> join(const Restrictions& lhs, const Restrictions& rhs);

void join_inplace(Restrictions& lhs, const Restrictions& rhs);

}  // namespace legate
