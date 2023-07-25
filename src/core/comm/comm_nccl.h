/* Copyright 2022 NVIDIA Corporation
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

#include "legion.h"

namespace legate::detail {
class Library;
}  // namespace legate::detail

namespace legate::comm::nccl {

void register_tasks(Legion::Runtime* runtime, const detail::Library* core_library);

bool needs_barrier();

void register_factory(const detail::Library* core_library);

}  // namespace legate::comm::nccl
