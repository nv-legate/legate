/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace scalarout {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

struct Copy : public legate::LegateTask<Copy> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0).data();
    auto output = context.output(0).data();
    auto shape  = output.shape<1>();
    if (shape.empty()) {
      return;
    }
    auto out_acc = output.write_accessor<std::int64_t, 1>();
    auto in_acc  = input.read_accessor<std::int64_t, 1>();
    out_acc[0]   = in_acc[0];
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_scalar_out";

  static void registration_callback(legate::Library library) { Copy::register_variants(library); }
};

class ScalarOut : public RegisterOnceFixture<Config> {};

void test_scalar_out()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  const legate::Shape extents{16};
  auto input  = runtime->create_store(extents, legate::int64(), false /* optimize_scalar */);
  auto output = runtime->create_store(legate::Scalar{int64_t{0}});

  runtime->issue_fill(input, legate::Scalar{int64_t{123}});

  {
    auto sliced   = input.slice(0, legate::Slice{2, 3});
    auto task     = runtime->create_task(library, Copy::TASK_CONFIG.task_id());
    auto part_in  = task.add_input(sliced);
    auto part_out = task.add_output(output);
    task.add_constraint(legate::align(part_in, part_out));
    runtime->submit(std::move(task));
  }

  auto p_out = output.get_physical_store();
  auto acc   = p_out.read_accessor<std::int64_t, 1>();
  EXPECT_EQ(acc[0], 123);
}

}  // namespace

TEST_F(ScalarOut, All) { test_scalar_out(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace scalarout
