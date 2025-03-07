/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_inout {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::string_view LIBRARY_NAME = "test_inout";

class TesterMapper : public legate::mapping::Mapper {
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    std::vector<legate::mapping::StoreMapping> mappings;
    auto inputs  = task.inputs();
    auto outputs = task.outputs();

    mappings.reserve(inputs.size() + outputs.size());
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

  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    return legate::Scalar{};
  }

  std::optional<std::size_t> allocation_pool_size(
    const legate::mapping::Task& /*task*/, legate::mapping::StoreTarget /*memory_kind*/) override
  {
    return std::nullopt;
  }
};

struct InoutTask : public legate::LegateTask<InoutTask> {
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto shape  = output.shape<2>();

    if (shape.empty()) {
      return;
    }

    auto acc = output.read_write_accessor<std::int64_t, 2>(shape);
    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      auto p = *it;
      EXPECT_EQ(acc[p], 123);
      acc[*it] = (p[0] + 1) + (p[1] + 1) * 1000;
    }
  }
};

class InOutUnit : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(
      LIBRARY_NAME, legate::ResourceConfig{}, std::make_unique<TesterMapper>(), {}, &created);
    if (created) {
      InoutTask::register_variants(library);
    }
  }
};

void test_inout()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(LIBRARY_NAME);

  const auto stores = {
    runtime->create_store(legate::Shape{10, 10}, legate::int64()),
    runtime->create_store(legate::Scalar{int64_t{0}}, legate::Shape{1, 1}),
  };

  for (auto& store : stores) {
    runtime->issue_fill(store, legate::Scalar{std::int64_t{123}});

    auto task     = runtime->create_task(library, InoutTask::TASK_ID);
    auto in_part  = task.add_input(store);
    auto out_part = task.add_output(store);
    task.add_constraint(legate::align(in_part, out_part));
    runtime->submit(std::move(task));

    auto p_out = store.get_physical_store();
    auto acc   = p_out.read_accessor<std::int64_t, 2>();
    auto shape = p_out.shape<2>();
    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      auto p = *it;
      EXPECT_EQ(acc[p], (p[0] + 1) + ((p[1] + 1) * 1000));
    }
  }
}

}  // namespace

TEST_F(InOutUnit, All) { test_inout(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace test_inout
