/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/attach.h>

#include <legate/operation/detail/operation.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace operation_attach_test {

using AttachUnit = DefaultFixture;

TEST_F(AttachUnit, Kind)
{
  constexpr auto unique_id = std::uint64_t{1};
  constexpr auto dim       = std::uint32_t{1};
  const auto region_field  = legate::InternalSharedPtr<legate::detail::LogicalRegionField>{};
  const auto allocation    = legate::InternalSharedPtr<legate::detail::ExternalAllocation>{};
  const auto ordering      = legate::InternalSharedPtr<legate::mapping::detail::DimOrdering>{};

  const auto attach = legate::detail::Attach{unique_id, region_field, dim, allocation, ordering};

  ASSERT_EQ(attach.kind(), legate::detail::Operation::Kind::ATTACH);
  ASSERT_FALSE(attach.needs_flush());
  ASSERT_FALSE(attach.needs_partitioning());
}

}  // namespace operation_attach_test
