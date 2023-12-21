/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace manual_task_test {

using ManualTask = DefaultFixture;

namespace {

constexpr const char* library_name     = "test_manual_task_proj";
constexpr const size_t DIM_EXTENT      = 32;
constexpr const size_t N_TILES_PER_DIM = 2;

}  // namespace

struct ProjTesterTask : public legate::LegateTask<ProjTesterTask> {
  static const int32_t TASK_ID = 1;
  static void cpu_variant(legate::TaskContext context)
  {
    auto task_index = context.get_task_index();
    auto row_wise   = context.input(0).shape<2>();
    auto col_wise   = context.input(1).shape<2>();

    EXPECT_EQ(row_wise.lo[0], DIM_EXTENT / N_TILES_PER_DIM * task_index[0]);
    EXPECT_EQ(row_wise.hi[0], DIM_EXTENT / N_TILES_PER_DIM * (task_index[0] + 1) - 1);
    EXPECT_EQ(row_wise.lo[1], int64_t{0});
    EXPECT_EQ(row_wise.hi[1], int64_t{DIM_EXTENT - 1});

    EXPECT_EQ(col_wise.lo[0], int64_t{0});
    EXPECT_EQ(col_wise.hi[0], int64_t{DIM_EXTENT - 1});
    EXPECT_EQ(col_wise.lo[1], DIM_EXTENT / N_TILES_PER_DIM * task_index[1]);
    EXPECT_EQ(col_wise.hi[1], DIM_EXTENT / N_TILES_PER_DIM * (task_index[1] + 1) - 1);
  }
};

TEST_F(ManualTask, Proj)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  ProjTesterTask::register_variants(library);

  auto store = runtime->create_store(legate::Shape{DIM_EXTENT, DIM_EXTENT}, legate::int64());
  runtime->issue_fill(store, legate::Scalar{int64_t{1}});

  auto row_wise = store.partition_by_tiling({DIM_EXTENT / N_TILES_PER_DIM, DIM_EXTENT});
  auto col_wise = store.partition_by_tiling({DIM_EXTENT, DIM_EXTENT / N_TILES_PER_DIM});

  // With a launch shape
  {
    auto task = runtime->create_task(
      library, ProjTesterTask::TASK_ID, legate::Shape{N_TILES_PER_DIM, N_TILES_PER_DIM});
    task.add_input(row_wise, legate::SymbolicPoint{legate::dimension(0), legate::constant(0)});
    task.add_input(col_wise, legate::SymbolicPoint{legate::constant(0), legate::dimension(1)});
    runtime->submit(std::move(task));
  }
  // With a launch domain
  {
    legate::Domain launch_domain{legate::Point<2>{1, 1},
                                 legate::Point<2>{N_TILES_PER_DIM - 1, N_TILES_PER_DIM - 1}};
    auto task = runtime->create_task(library, ProjTesterTask::TASK_ID, launch_domain);
    task.add_input(row_wise, legate::SymbolicPoint{legate::dimension(0), legate::constant(0)});
    task.add_input(col_wise, legate::SymbolicPoint{legate::constant(0), legate::dimension(1)});
    runtime->submit(std::move(task));
  }
}

}  // namespace manual_task_test
