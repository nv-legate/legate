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

namespace unbound_nullable_array_test {

using UnboundNullableArray = DefaultFixture;

constexpr const char* library_name = "test_unbound_nullable_array_test";

struct Initialize : public legate::LegateTask<Initialize> {
  static const std::int32_t TASK_ID = 0;
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

TEST_F(UnboundNullableArray, Bug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  Initialize::register_variants(library);

  auto task = runtime->create_task(library, Initialize::TASK_ID);
  task.add_output(runtime->create_array(legate::int64(), 1, true /*nullable*/));
  task.add_output(runtime->create_array(legate::list_type(legate::int64()), 1, true /*nullable*/));
  task.add_output(
    runtime->create_array(legate::struct_type(true, legate::int64()), 1, true /*nullable*/));
  runtime->submit(std::move(task));
}

}  // namespace unbound_nullable_array_test
