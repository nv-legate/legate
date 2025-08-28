/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace unbound_nullable_array_test {

struct Initialize : public legate::LegateTask<Initialize> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    auto arr_prim   = context.output(0);
    auto arr_list   = context.output(1).as_list_array();
    auto arr_struct = context.output(2);

    arr_prim.data().bind_empty_data();
    arr_prim.null_mask().bind_empty_data();

    arr_list.descriptor().data().bind_empty_data();
    arr_list.descriptor().null_mask().bind_empty_data();
    arr_list.vardata().data().bind_empty_data();

    arr_struct.child(0).data().bind_empty_data();
    arr_struct.null_mask().bind_empty_data();
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_unbound_nullable_array_test";

  static void registration_callback(legate::Library library)
  {
    Initialize::register_variants(library);
  }
};

class UnboundNullableArray : public RegisterOnceFixture<Config> {};

TEST_F(UnboundNullableArray, Bug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto task = runtime->create_task(library, Initialize::TASK_CONFIG.task_id());
  task.add_output(runtime->create_array(legate::int64(), 1, true /*nullable*/));
  task.add_output(runtime->create_array(legate::list_type(legate::int64()), 1, true /*nullable*/));
  task.add_output(
    runtime->create_array(legate::struct_type(true, legate::int64()), 1, true /*nullable*/));
  runtime->submit(std::move(task));
}

TEST_F(UnboundNullableArray, Bug2)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto task = runtime->create_task(library, Initialize::TASK_CONFIG.task_id());
  task.add_output(runtime->create_array(legate::int64(), 1, true /*nullable*/));
  task.add_output(runtime->create_array(legate::list_type(legate::int64()), 1, true /*nullable*/));
  task.add_output(
    runtime->create_array(legate::struct_type(true, legate::int64()), 1, true /*nullable*/));
  // Add a dummy array argument to get the task parallelized
  task.add_output(runtime->create_array(legate::Shape{4}, legate::int64(), true /*nullable*/));
  runtime->submit(std::move(task));
}

}  // namespace unbound_nullable_array_test
