/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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
