/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <legate/utilities/detail/store_iterator_cache.h>

namespace legate::detail {

template <typename T>
template <typename U>
typename StoreIteratorCache<T>::container_type& StoreIteratorCache<T>::operator()(const U& array)
{
  cache_.clear();
  array.populate_stores(cache_);
  return cache_;
}

template <typename T>
template <typename U>
const typename StoreIteratorCache<T>::container_type& StoreIteratorCache<T>::operator()(
  const U& array) const
{
  cache_.clear();
  array.populate_stores(cache_);
  return cache_;
}

}  // namespace legate::detail
