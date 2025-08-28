/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace manual_task_test {

namespace {

constexpr const std::size_t DIM_EXTENT      = 32;
constexpr const std::size_t N_TILES_PER_DIM = 2;

struct ProjTesterTask : public legate::LegateTask<ProjTesterTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto&& task_index = context.get_task_index();
    auto row_wise     = context.input(0).shape<2>();
    auto col_wise     = context.input(1).shape<2>();

    EXPECT_EQ(row_wise.lo[0], DIM_EXTENT / N_TILES_PER_DIM * task_index[0]);
    EXPECT_EQ(row_wise.hi[0], (DIM_EXTENT / N_TILES_PER_DIM * (task_index[0] + 1)) - 1);
    EXPECT_EQ(row_wise.lo[1], std::int64_t{0});
    EXPECT_EQ(row_wise.hi[1], std::int64_t{DIM_EXTENT - 1});

    EXPECT_EQ(col_wise.lo[0], std::int64_t{0});
    EXPECT_EQ(col_wise.hi[0], std::int64_t{DIM_EXTENT - 1});
    EXPECT_EQ(col_wise.lo[1], DIM_EXTENT / N_TILES_PER_DIM * task_index[1]);
    EXPECT_EQ(col_wise.hi[1], (DIM_EXTENT / N_TILES_PER_DIM * (task_index[1] + 1)) - 1);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_manual_task_proj";

  static void registration_callback(legate::Library library)
  {
    ProjTesterTask::register_variants(library);
  }
};

class ManualTaskWithProj : public RegisterOnceFixture<Config> {};

}  // namespace

TEST_F(ManualTaskWithProj, All)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto store = runtime->create_store(legate::Shape{DIM_EXTENT, DIM_EXTENT}, legate::int64());
  runtime->issue_fill(store, legate::Scalar{int64_t{1}});

  auto row_wise = store.partition_by_tiling({DIM_EXTENT / N_TILES_PER_DIM, DIM_EXTENT});
  auto col_wise = store.partition_by_tiling({DIM_EXTENT, DIM_EXTENT / N_TILES_PER_DIM});

  // With a launch shape
  {
    auto task =
      runtime->create_task(library,
                           ProjTesterTask::TASK_CONFIG.task_id(),
                           legate::tuple<std::uint64_t>{N_TILES_PER_DIM, N_TILES_PER_DIM});
    task.add_input(row_wise, legate::SymbolicPoint{legate::dimension(0), legate::constant(0)});
    task.add_input(col_wise, legate::SymbolicPoint{legate::constant(0), legate::dimension(1)});
    runtime->submit(std::move(task));
  }
  // With a launch domain
  {
    const legate::Domain launch_domain{legate::Point<2>{1, 1},
                                       legate::Point<2>{N_TILES_PER_DIM - 1, N_TILES_PER_DIM - 1}};
    auto task = runtime->create_task(library, ProjTesterTask::TASK_CONFIG.task_id(), launch_domain);
    task.add_input(row_wise, legate::SymbolicPoint{legate::dimension(0), legate::constant(0)});
    task.add_input(col_wise, legate::SymbolicPoint{legate::constant(0), legate::dimension(1)});
    runtime->submit(std::move(task));
  }
}

}  // namespace manual_task_test
