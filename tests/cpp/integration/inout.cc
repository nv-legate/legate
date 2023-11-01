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

#include <gtest/gtest.h>

#include "legate.h"
#include "utilities/utilities.h"

namespace inout {

using Integration = DefaultFixture;

enum TaskOpCode {
  INOUT = 1,
};

static const char* library_name = "test_inout";

class TesterMapper : public legate::mapping::Mapper {
  void set_machine(const legate::mapping::MachineQueryInterface* machine) override {}
  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override
  {
    return options.front();
  }
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    std::vector<legate::mapping::StoreMapping> mappings;
    auto inputs  = task.inputs();
    auto outputs = task.outputs();
    for (auto& input : inputs) {
      mappings.push_back(
        legate::mapping::StoreMapping::default_mapping(input.data(), options.front()));
    }
    for (auto& output : outputs) {
      mappings.push_back(
        legate::mapping::StoreMapping::default_mapping(output.data(), options.front()));
    }
    return mappings;
  }
  legate::Scalar tunable_value(legate::TunableID tunable_id) override { return legate::Scalar{}; }
};

struct InoutTask : public legate::LegateTask<InoutTask> {
  static const int32_t TASK_ID = INOUT;
  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto shape  = output.shape<2>();

    if (shape.empty()) return;

    auto acc = output.read_write_accessor<int64_t, 2>(shape);
    for (legate::PointInRectIterator<2> it(shape); it.valid(); ++it) {
      auto p = *it;
      EXPECT_EQ(acc[p], 123);
      acc[*it] = (p[0] + 1) + (p[1] + 1) * 1000;
    }
  }
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(
    library_name, legate::ResourceConfig{}, std::make_unique<TesterMapper>());
  InoutTask::register_variants(library);
}

void test_inout()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  std::vector<legate::LogicalStore> stores{
    runtime->create_store(legate::Shape{10, 10}, legate::int64()),
    runtime->create_store(legate::Scalar{int64_t{0}}, legate::Shape{1, 1}),
  };

  for (auto& store : stores) {
    runtime->issue_fill(store, legate::Scalar(int64_t(123)));

    auto task     = runtime->create_task(library, INOUT);
    auto in_part  = task.add_input(store);
    auto out_part = task.add_output(store);
    task.add_constraint(legate::align(in_part, out_part));
    runtime->submit(std::move(task));

    auto p_out = store.get_physical_store();
    auto acc   = p_out.read_accessor<int64_t, 2>();
    auto shape = p_out.shape<2>();
    for (legate::PointInRectIterator<2> it(shape); it.valid(); ++it) {
      auto p = *it;
      EXPECT_EQ(acc[p], (p[0] + 1) + (p[1] + 1) * 1000);
    }
  }
}

TEST_F(Integration, InOut)
{
  register_tasks();
  test_inout();
}

}  // namespace inout
