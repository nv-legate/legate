/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/partition_placement_info.h>

#include <legate/data/partition_placement_info.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace partition_placement_info_test {

using PartitionPlacementTest = DefaultFixture;

TEST_F(PartitionPlacementTest, ConstructAndAccess)
{
  const legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> color = {0, 1, 2};
  const std::uint32_t node_id                                            = 5;
  const legate::mapping::StoreTarget memory_type = legate::mapping::StoreTarget::SYSMEM;

  auto detail_ptr =
    legate::make_internal_shared<legate::detail::PartitionPlacement>(color, node_id, memory_type);
  const legate::PartitionPlacement placement{std::move(detail_ptr)};

  auto result_color = placement.partition_color();

  ASSERT_EQ(result_color.size(), 3);
  ASSERT_THAT(result_color, ::testing::ElementsAre(0, 1, 2));
  ASSERT_EQ(placement.node_id(), node_id);
  ASSERT_EQ(placement.memory_type(), memory_type);
}

TEST_F(PartitionPlacementTest, ConstructEmptyColor)
{
  const legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> color = {};
  const std::uint32_t node_id                                            = 0;
  const legate::mapping::StoreTarget memory_type = legate::mapping::StoreTarget::FBMEM;

  auto detail_ptr =
    legate::make_internal_shared<legate::detail::PartitionPlacement>(color, node_id, memory_type);
  const legate::PartitionPlacement placement{std::move(detail_ptr)};
  auto result_color = placement.partition_color();

  ASSERT_THAT(result_color, ::testing::IsEmpty());
  ASSERT_EQ(placement.node_id(), node_id);
  ASSERT_EQ(placement.memory_type(), memory_type);
}

using PartitionPlacementInfoTest = DefaultFixture;

TEST_F(PartitionPlacementInfoTest, ConstructEmpty)
{
  std::vector<legate::detail::PartitionPlacement> mappings{};
  auto detail_ptr =
    legate::make_internal_shared<legate::detail::PartitionPlacementInfo>(std::move(mappings));
  const legate::PartitionPlacementInfo info{std::move(detail_ptr)};

  auto result_mappings = info.placements();

  ASSERT_EQ(result_mappings.size(), 0);

  legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> test_color = {0};
  auto mapping = info.get_placement_for_color(test_color);

  ASSERT_FALSE(mapping.has_value());
}

TEST_F(PartitionPlacementInfoTest, MultipleMappings)
{
  std::vector<legate::detail::PartitionPlacement> mappings{};

  mappings.emplace_back(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{0, 0},
                        1,
                        legate::mapping::StoreTarget::SYSMEM);
  mappings.emplace_back(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{0, 1},
                        2,
                        legate::mapping::StoreTarget::FBMEM);
  mappings.emplace_back(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 0},
                        3,
                        legate::mapping::StoreTarget::SOCKETMEM);
  mappings.emplace_back(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1},
                        4,
                        legate::mapping::StoreTarget::ZCMEM);

  auto detail_ptr =
    legate::make_internal_shared<legate::detail::PartitionPlacementInfo>(std::move(mappings));
  const legate::PartitionPlacementInfo info{std::move(detail_ptr)};

  auto result_mappings = info.placements();

  ASSERT_EQ(result_mappings.size(), 4);

  const std::vector<std::tuple<legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>,
                               std::uint32_t,
                               legate::mapping::StoreTarget>>
    expected = {
      {{0, 0}, 1, legate::mapping::StoreTarget::SYSMEM},
      {{0, 1}, 2, legate::mapping::StoreTarget::FBMEM},
      {{1, 0}, 3, legate::mapping::StoreTarget::SOCKETMEM},
      {{1, 1}, 4, legate::mapping::StoreTarget::ZCMEM},
    };

  for (const auto& [color, expected_node, expected_memory] : expected) {
    auto found_mapping = info.get_placement_for_color(color);

    if (found_mapping) {
      ASSERT_EQ(found_mapping->node_id(), expected_node);
      ASSERT_EQ(found_mapping->memory_type(), expected_memory);
    } else {
      FAIL() << "Mapping for color not found";
    }
  }

  legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> nonexistent_color = {2, 2};
  auto not_found = info.get_placement_for_color(nonexistent_color);

  ASSERT_FALSE(not_found.has_value());
}

TEST_F(PartitionPlacementInfoTest, ToString)
{
  std::vector<legate::detail::PartitionPlacement> mappings;
  mappings.emplace_back(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{0},
                        1,
                        legate::mapping::StoreTarget::SYSMEM);
  mappings.emplace_back(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{0, 1},
                        2,
                        legate::mapping::StoreTarget::FBMEM);

  auto detail_ptr =
    legate::make_internal_shared<legate::detail::PartitionPlacementInfo>(std::move(mappings));
  const legate::PartitionPlacementInfo info{std::move(detail_ptr)};

  const std::string result = info.to_string();

  ASSERT_THAT(result, ::testing::HasSubstr("Partition Color"));
  ASSERT_THAT(result, ::testing::HasSubstr("Node"));
  ASSERT_THAT(result, ::testing::HasSubstr("Memory"));
  ASSERT_THAT(result, ::testing::HasSubstr("SYSMEM"));
  ASSERT_THAT(result, ::testing::HasSubstr("1"));
  ASSERT_THAT(result, ::testing::HasSubstr("[0]"));
  ASSERT_THAT(result, ::testing::HasSubstr("2"));
  ASSERT_THAT(result, ::testing::HasSubstr("[0, 1]"));
  ASSERT_THAT(result, ::testing::HasSubstr("FBMEM"));
}

}  // namespace partition_placement_info_test
