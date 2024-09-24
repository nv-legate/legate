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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

// Note on how to run this test properly:
//
// Since we don't have a mechanism to run different tests under different machine configurations
// yet, this test needs to be run with the following command to confirm the correct behavior:
//
//   ./test.py
//    --gtest-file "$LEGATE_ARCH"/cmake_build/tests/cpp/bin/tests_with_runtime
//    --gtest-filter=Redundant.Test --sysmem 32
//
// i.e., the test is designed to use only up to 16MB (50% of 32MB) of the system memory space.
//
namespace test_redundant {

namespace {

constexpr std::string_view LIBRARY_NAME = "test_redundant";
constexpr std::uint64_t EXT             = 1 << 10;

struct Tester : public legate::LegateTask<Tester> {
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    // Does nothing
  }
};

class LibraryMapper : public legate::mapping::Mapper {
  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& /*task*/,
    const std::vector<legate::mapping::TaskTarget>& options) override
  {
    return options.front();
  }
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    auto mappings = std::vector<legate::mapping::StoreMapping>{};
    for (auto&& input : task.inputs()) {
      auto mapping =
        legate::mapping::StoreMapping::default_mapping(input.data(), options.front(), true);
      mapping.policy().redundant = true;
      mappings.push_back(std::move(mapping));
    }
    return mappings;
  }
  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    return legate::Scalar{};
  }
};

class Redundant : public DefaultFixture {
  void SetUp() override
  {
    DefaultFixture::SetUp();
    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(
      LIBRARY_NAME, legate::ResourceConfig{}, std::make_unique<LibraryMapper>(), {}, &created);
    if (created) {
      Tester::register_variants(library);
    }
  }
};

void launch_task(const legate::LogicalStorePartition& part, bool read_only)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(LIBRARY_NAME);
  auto task    = runtime->create_task(library, Tester::TASK_ID, part.color_shape());
  if (read_only) {
    task.add_input(part);
  } else {
    task.add_output(part);
  }
  runtime->submit(std::move(task));
}

}  // namespace

TEST_F(Redundant, Test)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{EXT, EXT}, legate::int64());
  auto part1   = store.partition_by_tiling({EXT / 2, EXT / 2});
  auto part2   = store.partition_by_tiling({EXT / 4, EXT});
  auto part3   = store.partition_by_tiling({EXT, EXT / 4});

  launch_task(part1, false);
  launch_task(part2, true);
  // Wihtout a mapping fence, all reader tasks in theory can initiate their mapping before mappings
  // for other tasks finish and yield the memory space
  runtime->issue_mapping_fence();
  launch_task(part3, true);
}

}  // namespace test_redundant
