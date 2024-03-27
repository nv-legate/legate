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

#include "core/utilities/detail/ordered_set.h"

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
