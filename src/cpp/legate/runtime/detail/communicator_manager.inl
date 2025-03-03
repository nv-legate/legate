/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/communicator_manager.h>

namespace legate::detail {

template <class Desc>
mapping::detail::Machine CommunicatorFactory::CacheKey<Desc>::get_machine() const
{
  return mapping::detail::Machine({{target, range}});
}

template <class Desc>
bool CommunicatorFactory::CacheKey<Desc>::operator==(const CacheKey& other) const
{
  return desc == other.desc && target == other.target && range == other.range;
}

template <class Desc>
std::size_t CommunicatorFactory::CacheKey<Desc>::hash() const noexcept
{
  return hash_all(desc, target, range);
}

}  // namespace legate::detail
