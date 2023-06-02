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
#include "core/utilities/typedefs.h"

namespace legate {

using Shape = tuple<size_t>;

Domain to_domain(const Shape& shape);

DomainPoint to_domain_point(const Shape& shape);

Shape from_domain(const Domain& domain);

}  // namespace legate
