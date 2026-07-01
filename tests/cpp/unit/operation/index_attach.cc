/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/index_attach.h>

#include <legate/operation/detail/operation.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>
#include <vector>

namespace operation_index_attach_test {

using IndexAttachUnit = DefaultFixture;

TEST_F(IndexAttachUnit, Kind)
{
  constexpr auto unique_id = std::uint64_t{1};
  constexpr auto dim       = std::uint32_t{1};
  const auto region_field  = legate::InternalSharedPtr<legate::detail::LogicalRegionField>{};
  const auto subregions    = std::vector<Legion::LogicalRegion>{};
  const auto allocations =
    std::vector<legate::InternalSharedPtr<legate::detail::ExternalAllocation>>{};
  const auto ordering = legate::InternalSharedPtr<legate::mapping::detail::DimOrdering>{};

  const auto index_attach =
    legate::detail::IndexAttach{unique_id, region_field, dim, subregions, allocations, ordering};

  ASSERT_EQ(index_attach.kind(), legate::detail::Operation::Kind::INDEX_ATTACH);
  ASSERT_FALSE(index_attach.needs_flush());
  ASSERT_FALSE(index_attach.needs_partitioning());
}

}  // namespace operation_index_attach_test
