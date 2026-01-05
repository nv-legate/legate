/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/default_mapper.h>

namespace legate::mapping::detail {

inline std::vector<mapping::StoreMapping> DefaultMapper::store_mappings(
  const mapping::Task& /*task*/, const std::vector<StoreTarget>& /*options*/)
{
  return {};
}

inline Scalar DefaultMapper::tunable_value(TunableID /*tunable_id*/)
{
  LEGATE_ABORT("Should not be called!");
  return Scalar{0};
}

inline std::optional<std::size_t> DefaultMapper::allocation_pool_size(const Task& /*task*/,
                                                                      StoreTarget /*memory_kind*/)
{
  return std::nullopt;
}

}  // namespace legate::mapping::detail
