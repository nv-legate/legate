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

#include <stdint.h>
#include <optional>

/**
 * @file
 * @brief A simple slice class that has the same semantics as Python's
 */

namespace legate {

/**
 * @ingroup data
 * @brief A slice descriptor
 *
 * legate::Slice behaves similarly to how the slice in Python does, and has different semantics
 * from std::slice.
 */
struct Slice {
  static constexpr std::nullopt_t OPEN = std::nullopt;

  Slice(std::optional<int64_t> _start = OPEN, std::optional<int64_t> _stop = OPEN)
    : start(_start), stop(_stop)
  {
  }

  std::optional<int64_t> start{OPEN};
  std::optional<int64_t> stop{OPEN};
};

}  // namespace legate
