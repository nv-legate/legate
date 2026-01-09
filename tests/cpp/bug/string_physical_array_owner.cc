/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_string_physical_array_owner {

namespace {

class InitStringTask : public legate::LegateTask<InitStringTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}
      .with_signature(legate::TaskSignature{}.outputs(1))
      .with_variant_options(legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void InitStringTask::cpu_variant(legate::TaskContext context)
{
  const auto array        = context.output(0);
  const auto string_array = array.as_string_array();
  const auto chars        = string_array.chars().data();
  constexpr auto DIM      = 1;
  constexpr auto SIZE     = 10;
  const auto buf          = chars.create_output_buffer<std::int8_t>(legate::Point<DIM>{SIZE},
                                                           /* bind_buffer */ true);

  for (auto it = legate::PointInRectIterator<DIM>{buf.get_bounds()}; it.valid(); ++it) {
    buf[*it] = 0;
  }
  string_array.ranges().data().bind_empty_data();
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_string_physical_array_owner";

  static void registration_callback(legate::Library library)
  {
    InitStringTask::register_variants(library);
  }
};

}  // namespace

class StringPhysicalArrayOwner : public RegisterOnceFixture<Config> {};

// See https://github.com/nv-legate/legate.internal/issues/2239
TEST_F(StringPhysicalArrayOwner, Chars)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  const auto array    = runtime->create_array(legate::string_type());

  {
    auto task = runtime->create_task(lib, InitStringTask::TASK_CONFIG.task_id());

    task.add_output(array);
    runtime->submit(std::move(task));
  }

  const auto str   = array.get_physical_array().as_string_array();
  const auto chars = str.chars();

  // This line should throw with "Data store of a nested array cannot be retrieved" if we
  // aren't propagating the owners correctly.
  ASSERT_NO_THROW(std::ignore = chars.data());
}

TEST_F(StringPhysicalArrayOwner, Ranges)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  const auto array    = runtime->create_array(legate::string_type());

  {
    auto task = runtime->create_task(lib, InitStringTask::TASK_CONFIG.task_id());

    task.add_output(array);
    runtime->submit(std::move(task));
  }

  const auto str    = array.get_physical_array().as_string_array();
  const auto ranges = str.ranges();

  // This line should throw with "Data store of a nested array cannot be retrieved" if we
  // aren't propagating the owners correctly.
  ASSERT_NO_THROW(std::ignore = ranges.data());
}

}  // namespace test_string_physical_array_owner
