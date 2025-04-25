/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_offload_array {

namespace {

class FillSingletonRectsTask : public legate::LegateTask<FillSingletonRectsTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto store       = context.output(0).data();
    auto shape       = store.shape<1>();
    auto acc         = store.write_accessor<legate::Rect<1>, 1, true>(shape);
    std::int64_t idx = 0;
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      acc[*it] = legate::Rect<1>{idx, idx + 1};
      ++idx;
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "TEST_OFFLOAD_ARRAY";
  static void registration_callback(legate::Library library)
  {
    FillSingletonRectsTask::register_variants(library);
  }
};

constexpr auto ARBITRARY_SIZE  = 5;
constexpr auto ARBITRARY_VALUE = 42;

}  // namespace

// These tests are checking that offloading different types of LogicalARrays doesn't throw an
// exception about mismatched signature for the offload task.
// See https://github.com/nv-legate/legate.internal/issues/1930

class OffloadArray : public RegisterOnceFixture<Config> {};

TEST_F(OffloadArray, NormalArray)
{
  auto* runtime = legate::Runtime::get_runtime();
  auto array    = runtime->create_array(legate::Shape{ARBITRARY_SIZE}, legate::int32());
  runtime->issue_fill(array, legate::Scalar{std::int32_t{1}});
  ASSERT_NO_THROW(array.offload_to(legate::mapping::StoreTarget::SYSMEM));
}

TEST_F(OffloadArray, NullableArray)
{
  auto* runtime = legate::Runtime::get_runtime();
  auto array =
    runtime->create_array(legate::Shape{ARBITRARY_SIZE}, legate::int32(), /* nullable=*/true);
  runtime->issue_fill(array, legate::Scalar{std::int32_t{1}});
  ASSERT_NO_THROW(array.offload_to(legate::mapping::StoreTarget::SYSMEM));
}

TEST_F(OffloadArray, ListArray)
{
  auto* runtime = legate::Runtime::get_runtime();
  auto library  = runtime->find_library(Config::LIBRARY_NAME);

  auto descriptor = runtime->create_array(legate::Shape{ARBITRARY_SIZE}, legate::rect_type(1));
  auto task       = runtime->create_task(library, FillSingletonRectsTask::TASK_CONFIG.task_id());
  task.add_output(descriptor);
  runtime->submit(std::move(task));

  auto vardata = runtime->create_array(legate::Shape{ARBITRARY_SIZE}, legate::int32());
  runtime->issue_fill(vardata, legate::Scalar{std::int32_t{ARBITRARY_VALUE}});

  auto array = runtime->create_list_array(descriptor, vardata);
  ASSERT_NO_THROW(array.offload_to(legate::mapping::StoreTarget::SYSMEM));
}

}  // namespace test_offload_array
