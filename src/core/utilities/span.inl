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

#include "core/utilities/span.h"

#include <cassert>

namespace legate {

template <typename T>
Span<T>::Span(T* data, std::size_t size) : data_{data}, size_{size}
{
}

template <typename T>
std::size_t Span<T>::size() const
{
  return size_;
}

template <typename T>
decltype(auto) Span<T>::operator[](std::size_t pos) const
{
  assert(pos < size_);
  return data_[pos];
}

template <typename T>
const T* Span<T>::begin() const
{
  return data_;
}

template <typename T>
const T* Span<T>::end() const
{
  return data_ + size_;
}

template <typename T>
Span<T> Span<T>::subspan(std::size_t off)
{
  assert(off <= size_);
  return {data_ + off, size_ - off};
}

template <typename T>
const T* Span<T>::ptr() const
{
  return data_;
}

}  // namespace legate
