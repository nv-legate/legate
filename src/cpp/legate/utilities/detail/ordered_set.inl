/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/ordered_set.h>

namespace legate {

template <typename T>
void ordered_set<T>::insert(const T& value)
{
  if (element_set_.find(value) != element_set_.end()) {
    return;
  }
  elements_.emplace_back(value);
  element_set_.insert(value);
}

template <typename T>
void ordered_set<T>::insert(T&& value)
{
  if (element_set_.find(value) != element_set_.end()) {
    return;
  }
  elements_.emplace_back(value);
  element_set_.insert(std::move(value));
}

template <typename T>
const std::vector<T>& ordered_set<T>::elements() const
{
  return elements_;
}

}  // namespace legate
