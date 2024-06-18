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

#include "core/runtime/detail/region_manager.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace region_manager {

class TesterTask : public legate::LegateTask<TesterTask> {
 public:
  static constexpr std::int32_t TASK_ID = 0;

  static void cpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    for (auto&& output : outputs) {
      auto store = output.data();
      if (store.is_unbound_store()) {
        store.bind_empty_data();
      }
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_region_manager";
  static void registration_callback(legate::Library library)
  {
    auto options =
      legate::VariantOptions{}.with_return_size(8192);  // NOLINT(readability-magic-numbers)
    TesterTask::register_variants(library, {{LEGATE_CPU_VARIANT, options}});
  }
};

class RegionManager : public RegisterOnceFixture<Config> {};

TEST_F(RegionManager, Normal)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, 0);

  std::vector<legate::LogicalStore> stores;
  for (std::uint32_t idx = 0; idx < legate::detail::RegionManager::MAX_NUM_FIELDS * 2; ++idx) {
    constexpr auto SHAPE_SIZE = 10;
    auto store                = runtime->create_store(legate::Shape{SHAPE_SIZE}, legate::int64());

    task.add_output(store);
    stores.emplace_back(std::move(store));
  }
  runtime->submit(std::move(task));
}

TEST_F(RegionManager, Unbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, 0);

  std::vector<legate::LogicalStore> stores;
  std::vector<legate::Variable> parts;
  for (std::uint32_t idx = 0; idx < legate::detail::RegionManager::MAX_NUM_FIELDS * 2; ++idx) {
    auto store = runtime->create_store(legate::int64(), 1);
    auto part  = task.add_output(store);
    stores.push_back(store);
    parts.push_back(part);
  }
  for (std::uint32_t idx = 1; idx < parts.size(); ++idx) {
    task.add_constraint(legate::align(parts.front(), parts[idx]));
  }
  runtime->submit(std::move(task));
}

}  // namespace region_manager
