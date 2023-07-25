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
  if (map_.end() == finder) return false;
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

template <typename T>
void MultiSet<T>::clear()
{
  map_.clear();
}

}  // namespace legate
