/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace key_partition_invalidation {

using InvalidateKeyPartition = DefaultFixture;

namespace {

constexpr std::string_view LIBRARY_NAME = "test_key_partition_invalidation";
constexpr legate::coord_t EXT           = 20;
constexpr legate::coord_t NUM_TILES     = 4;
constexpr legate::coord_t TILE_SIZE     = EXT / NUM_TILES;

}  // namespace

struct InitRanges : public legate::LegateTask<InitRanges> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0);
    auto shape  = output.shape<1>();
    auto acc    = output.data().write_accessor<legate::Rect<1>, 1>();
    for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
      auto idx = legate::coord_t{*it};
      acc[*it] = legate::Rect<1>{idx * TILE_SIZE, ((idx + 1) * TILE_SIZE) - 1};
    }
  }
};

struct Checker : public legate::LegateTask<Checker> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

// This test case exercises a scenario where an image partition cached as a key outlives its
// function store.
TEST_F(InvalidateKeyPartition, Bug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(LIBRARY_NAME);
  InitRanges::register_variants(library);
  Checker::register_variants(library);

  auto range = runtime->create_store(legate::Shape{EXT}, legate::int64());

  {
    auto func      = runtime->create_store(legate::Shape{NUM_TILES}, legate::rect_type(1));
    auto init_task = runtime->create_task(library, InitRanges::TASK_CONFIG.task_id());
    init_task.add_output(func);
    runtime->submit(std::move(init_task));

    auto writer_task = runtime->create_task(library, Checker::TASK_CONFIG.task_id());
    auto part_func   = writer_task.add_input(func);
    auto part_range  = writer_task.add_output(range);
    writer_task.add_constraint(legate::image(part_func, part_range));
    runtime->submit(std::move(writer_task));
  }

  // At this point, the func store is destroyed and its backing storage is also discarded.
  // Furthermore, the image partition that refers to the func store is also evicted from the
  // partition manager. However, if that image partition is cached somewhere else (e.g., as a key
  // partition in case of GH 2589), the partition can be used again in the downstream task to
  // construct a Legion partition, leading to an uninitialized access.
  {
    auto reader      = runtime->create_store(legate::Shape{EXT}, legate::int64());
    auto reader_task = runtime->create_task(library, Checker::TASK_CONFIG.task_id());
    reader_task.add_input(range);
    runtime->submit(std::move(reader_task));
  }
}

}  // namespace key_partition_invalidation
