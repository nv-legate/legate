/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_array_fill_list_test {

namespace {

class FillListTask : public legate::LegateTask<FillListTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class CheckListTask : public legate::LegateTask<CheckListTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  static void cpu_variant(legate::TaskContext);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_fill_list_physical_array";

  static void registration_callback(legate::Library library)
  {
    FillListTask::register_variants(library);
    CheckListTask::register_variants(library);
  }
};

class FillListPhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class NullableFillListArrayTest : public FillListPhysicalArrayUnit,
                                  public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(FillListPhysicalArrayUnit,
                         NullableFillListArrayTest,
                         ::testing::Values(true, false));

/*static*/ void FillListTask::cpu_variant(legate::TaskContext context)
{
  auto array                        = context.output(0);
  auto unbound                      = context.scalar(1).value<bool>();
  auto list_array                   = array.as_list_array();
  auto descriptor_store             = list_array.descriptor().data();
  auto vardata_store                = list_array.vardata().data();
  static constexpr std::int32_t DIM = 1;

  ASSERT_NO_THROW(static_cast<void>(
    vardata_store.create_output_buffer<std::int64_t, DIM>(legate::Point<DIM>{10}, true)));
  if (unbound) {
    ASSERT_NO_THROW(descriptor_store.bind_empty_data());
  }

  auto nullable = context.scalar(0).value<bool>();

  if (nullable) {
    auto null_mask = array.null_mask();

    if (null_mask.is_unbound_store()) {
      ASSERT_NO_THROW(null_mask.bind_empty_data());
    }
  }
}

/*static*/ void CheckListTask::cpu_variant(legate::TaskContext context)
{
  auto array                        = context.output(0);
  auto list_array                   = array.as_list_array();
  auto vardata_store                = list_array.vardata().data();
  static constexpr std::int32_t DIM = 1;
  auto vardata_shape                = vardata_store.shape<DIM>();

  if (!vardata_shape.empty()) {
    auto i          = 0;
    auto rw_vardata = vardata_store.read_write_accessor<std::int64_t, DIM>();

    for (legate::PointInRectIterator<DIM> it{vardata_shape}; it.valid(); ++it) {
      rw_vardata[*it] = i;
      i++;
    }
    i = 0;
    for (legate::PointInRectIterator<DIM> it{vardata_shape}; it.valid(); ++it) {
      ASSERT_EQ(rw_vardata[*it], i);
      i++;
    }
  }

  auto nullable = context.scalar(0).value<bool>();
  auto unbound  = context.scalar(1).value<bool>();

  if (nullable) {
    if (unbound) {
      return;
    }
    auto null_mask  = array.null_mask();
    auto mask_shape = null_mask.shape<DIM>();

    if (mask_shape.empty()) {
      return;
    }
    auto index   = 0;
    auto rw_mask = null_mask.read_write_accessor<bool, DIM>();

    for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
      rw_mask[*it] = (index % 2 == 0);
      index++;
    }
    index = 0;
    for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
      ASSERT_EQ(rw_mask[*it], (index % 2 == 0));
      index++;
    }
  }
}

void test_fill_list_array_task(legate::LogicalArray& logical_array, bool nullable, bool unbound)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto context  = runtime->find_library(Config::LIBRARY_NAME);
  auto arr_type = legate::list_type(legate::int64()).as_list_type();
  auto task     = runtime->create_task(context, FillListTask::TASK_ID);
  auto part     = task.declare_partition();

  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));

  task = runtime->create_task(context, CheckListTask::TASK_ID);
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(NullableFillListArrayTest, BoundListArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto logical_array  = runtime->create_array({3}, arr_type, nullable);
  test_fill_list_array_task(logical_array, nullable, false);
}

TEST_P(NullableFillListArrayTest, UnboundListArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto logical_array  = runtime->create_array(arr_type, 1, nullable);
  test_fill_list_array_task(logical_array, nullable, true);
}

}  // namespace physical_array_fill_list_test
