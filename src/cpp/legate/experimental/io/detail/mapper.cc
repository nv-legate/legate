/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/io/detail/mapper.h>

#include <legate/data/buffer.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/align.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/type_traits.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace legate::experimental::io::detail {

std::vector<mapping::StoreMapping> Mapper::store_mappings(const mapping::Task&,
                                                          const std::vector<mapping::StoreTarget>&)
{
  return {};
}

std::optional<std::size_t> Mapper::allocation_pool_size(const mapping::Task&, mapping::StoreTarget)
{
  return {};
}

legate::Scalar Mapper::tunable_value(TunableID)
{
  LEGATE_CHECK(false);
  return legate::Scalar{0};
}

}  // namespace legate::experimental::io::detail
