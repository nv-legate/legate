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

// Useful for IDEs
#include "core/utilities/multi_set.h"

namespace legate {

template <typename T>
void MultiSet<T>::add(const T& value)
{
  auto finder = map_.find(value);
  if (finder == map_.end())
    map_[value] = 1;
  else
    finder->second++;
}

template <typename T>
bool MultiSet<T>::remove(const T& value)
{
  auto finder = map_.find(value);
  assert(finder != map_.end());
  finder->second--;
  if (finder->second == 0) {
    map_.erase(finder);
    return true;
  }
  return false;
}

template <typename T>
bool MultiSet<T>::contains(const T& value) const
{
  return map_.contains(value);
}

}  // namespace legate
