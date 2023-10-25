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

namespace delinearize {

using Integration = DefaultFixture;

static const char* library_name = "test_delinearize";

enum TaskIDs {
  ARANGE = 0,
  COPY   = 1,
};

struct Arange : public legate::LegateTask<Arange> {
  static const int32_t TASK_ID = ARANGE;
  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto shape  = output.shape<1>();
    auto acc    = output.write_accessor<int64_t, 1>();
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      auto p = *it;
      acc[p] = p[0];
    }
  }
};

struct Copy : public legate::LegateTask<Copy> {
  static const int32_t TASK_ID = COPY;
  static void cpu_variant(legate::TaskContext context)
  {
    auto input   = context.input(0).data();
    auto output  = context.output(0).data();
    auto shape   = output.shape<3>();
    auto out_acc = output.write_accessor<int64_t, 3>();
    auto in_acc  = input.read_accessor<int64_t, 3>();
    for (legate::PointInRectIterator<3> it(shape); it.valid(); ++it) { out_acc[*it] = in_acc[*it]; }
  }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  Arange::register_variants(library);
  Copy::register_variants(library);
}

void test_delinearize()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  auto input = runtime->create_array(
    {
      16,
    },
    legate::int64());
  auto output = runtime->create_array({1, 8, 2}, legate::int64());

  {
    auto task = runtime->create_task(library, ARANGE);
    task.add_output(input);
    runtime->submit(std::move(task));
  }
  {
    auto transformed = input.promote(0, 1).delinearize(1, {8, 2});
    auto task        = runtime->create_task(library, COPY);
    auto part_in     = task.add_input(transformed);
    auto part_out    = task.add_output(output);
    task.add_constraint(legate::align(part_out, part_in));
    runtime->submit(std::move(task));
  }

  auto p_out = output.data().get_physical_store();
  auto acc   = p_out.read_accessor<int64_t, 3>();
  auto shape = p_out.shape<3>();
  for (legate::PointInRectIterator<3> it(shape); it.valid(); ++it) {
    auto p = *it;
    EXPECT_EQ(acc[p], 2 * p[1] + p[2]);
  }
}

TEST_F(Integration, Delinearize)
{
  register_tasks();
  test_delinearize();
}

}  // namespace delinearize
