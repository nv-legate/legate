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
  const auto [it, inserted] = element_set_.emplace(value);

  if (inserted) {
    elements_.emplace_back(*it);
  }
}

template <typename T>
void ordered_set<T>::insert(T&& value)
{
  const auto [it, inserted] = element_set_.emplace(std::move(value));

  if (inserted) {
    elements_.emplace_back(*it);
  }
}

template <typename T>
const std::vector<T>& ordered_set<T>::elements() const
{
  return elements_;
}

}  // namespace legate
