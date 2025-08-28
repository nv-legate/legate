/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace mixed_dim {

// NOLINTBEGIN(readability-magic-numbers)

struct Tester : public legate::LegateTask<Tester> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_TRUE(context.is_single_task());
    context.output(0).data().bind_empty_data();
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_mixed_dim";

  static void registration_callback(legate::Library library) { Tester::register_variants(library); }
};

class Partitioner : public RegisterOnceFixture<Config> {};

TEST_F(Partitioner, MixedDim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto test = [&runtime, &library](
                std::int32_t unbound_ndim, const auto& extents1, const auto& extents2) {
    auto unbound = runtime->create_store(legate::int32(), unbound_ndim);
    auto normal1 = runtime->create_store(extents1, legate::int64());
    auto normal2 = runtime->create_store(extents2, legate::float64());

    auto task = runtime->create_task(library, Tester::TASK_CONFIG.task_id());
    task.add_output(unbound);
    task.add_output(normal1);
    task.add_output(normal2);
    runtime->submit(std::move(task));
  };

  auto ty        = legate::int64();
  auto extents1d = legate::Shape{
    10,
  };
  auto extents2d = legate::Shape{10, 20};
  auto extents3d = legate::Shape{10, 20, 30};
  // Must-be-1D cases
  test(2, extents1d, extents2d);
  test(3, extents1d, extents2d);
  // Cases where normal stores lead to a single launch domain
  test(1, extents2d, extents2d);
  test(2, extents3d, extents3d);
  test(3, extents1d, extents1d);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace mixed_dim
