/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/cuda/detail/cuda_driver_api.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace inline_map {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

struct AdderTask : public legate::LegateTask<AdderTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto shape  = output.shape<1>();
    auto acc    = output.read_write_accessor<std::int64_t, 1>(shape);
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      acc[*it] += 1;
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_inline_map";
  static void registration_callback(legate::Library library)
  {
    AdderTask::register_variants(library);
  }
};

class InlineMap : public RegisterOnceFixture<Config> {};

void test_inline_map_future()
{
  auto runtime = legate::Runtime::get_runtime();
  auto l_store = runtime->create_store(legate::Shape{1}, legate::int64(), true /*optimize_scalar*/);
  auto p_store = l_store.get_physical_store();
  EXPECT_TRUE(p_store.is_future());
}

void test_inline_map_region_and_slice()
{
  auto runtime = legate::Runtime::get_runtime();
  auto root_ls = runtime->create_store(legate::Shape{5}, legate::int64());
  auto root_ps = root_ls.get_physical_store();
  EXPECT_FALSE(root_ps.is_future());
  auto slice_ls = root_ls.slice(0, legate::Slice(1));
  auto slice_ps = slice_ls.get_physical_store();
  EXPECT_FALSE(slice_ps.is_future());
  auto root_acc  = root_ps.write_accessor<std::int64_t, 1>();
  root_acc[2]    = 42;
  auto slice_acc = slice_ps.read_accessor<std::int64_t, 1>();
  EXPECT_EQ(slice_acc[1], 42);
}

void test_inline_map_and_task()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto l_store = runtime->create_store(legate::Shape{5}, legate::int64());
  {
    auto p_store = l_store.get_physical_store();
    auto acc     = p_store.write_accessor<std::int64_t, 1>();
    acc[2]       = 42;
  }
  auto task = runtime->create_task(context, AdderTask::TASK_CONFIG.task_id(), {1});
  task.add_input(l_store);
  task.add_output(l_store);
  runtime->submit(std::move(task));
  auto p_store = l_store.get_physical_store();
  auto acc     = p_store.read_accessor<std::int64_t, 1>();
  EXPECT_EQ(acc[2], 43);
}

void test_inline_map_region_gpu()
{
  if (legate::get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    GTEST_SKIP() << "Skipping the test when no GPU is found";
  }
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto l_store = runtime->create_store(legate::Shape{5}, legate::int64());

  {
    auto p_store = l_store.get_physical_store(legate::mapping::StoreTarget::FBMEM);
    auto acc     = p_store.write_accessor<std::int64_t, 1>();
    auto* ptr    = acc.ptr(2);
    auto value   = std::int64_t{42};

    const auto ctx    = legate::cuda::detail::AutoPrimaryContext{0};
    auto&& driver_api = legate::cuda::detail::get_cuda_driver_api();
    auto stream       = driver_api->stream_create(0);

    try {
      driver_api->mem_cpy_async(ptr, &value, sizeof(value), stream);
      driver_api->stream_synchronize(stream);
    } catch (...) {
      driver_api->stream_destroy(&stream);
      throw;
    }
    driver_api->stream_destroy(&stream);
  }
  {
    auto task = runtime->create_task(library, AdderTask::TASK_CONFIG.task_id(), {1});
    task.add_input(l_store);
    task.add_output(l_store);
    runtime->submit(std::move(task));
  }
  {
    auto p_store = l_store.get_physical_store();
    auto acc     = p_store.read_accessor<std::int64_t, 1>();
    EXPECT_EQ(acc[2], 43);
  }
}

}  // namespace

TEST_F(InlineMap, Future) { test_inline_map_future(); }

TEST_F(InlineMap, RegionAndSlice) { test_inline_map_region_and_slice(); }

TEST_F(InlineMap, WithTask) { test_inline_map_and_task(); }

TEST_F(InlineMap, RegionGPU) { test_inline_map_region_gpu(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace inline_map
