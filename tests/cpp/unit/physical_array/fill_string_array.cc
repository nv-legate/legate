/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_array_fill_string_test {

namespace {

class FillStringTask : public legate::LegateTask<FillStringTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class CheckStringTask : public legate::LegateTask<CheckStringTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  static void cpu_variant(legate::TaskContext);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_fill_string_physical_array";

  static void registration_callback(legate::Library library)
  {
    FillStringTask::register_variants(library);
    CheckStringTask::register_variants(library);
  }
};

class FillStringPhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class NullableFillStringArrayTest : public FillStringPhysicalArrayUnit,
                                    public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(FillStringPhysicalArrayUnit,
                         NullableFillStringArrayTest,
                         ::testing::Values(true, false));

/*static*/ void FillStringTask::cpu_variant(legate::TaskContext context)
{
  auto array                        = context.output(0);
  auto unbound                      = context.scalar(1).value<bool>();
  auto string_array                 = array.as_string_array();
  auto ranges_store                 = string_array.ranges().data();
  auto chars_store                  = string_array.chars().data();
  static constexpr std::int32_t DIM = 1;

  ASSERT_NO_THROW(static_cast<void>(
    chars_store.create_output_buffer<std::int8_t, 1>(legate::Point<DIM>{10}, true)));
  if (unbound) {
    ASSERT_NO_THROW(ranges_store.bind_empty_data());
  }
  auto nullable = context.scalar(0).value<bool>();

  if (nullable) {
    auto null_mask = array.null_mask();

    if (null_mask.is_unbound_store()) {
      ASSERT_NO_THROW(null_mask.bind_empty_data());
    }
  }
}

/*static*/ void CheckStringTask::cpu_variant(legate::TaskContext context)
{
  auto array                        = context.output(0);
  auto string_array                 = array.as_string_array();
  auto chars_store                  = string_array.chars().data();
  static constexpr std::int32_t DIM = 1;
  auto chars_shape                  = chars_store.shape<DIM>();

  if (!chars_shape.empty()) {
    auto i        = 0;
    auto rw_chars = chars_store.read_write_accessor<std::int8_t, DIM>();

    for (legate::PointInRectIterator<DIM> it{chars_shape}; it.valid(); ++it) {
      rw_chars[*it] = static_cast<std::int8_t>(i);
      i++;
    }
    i = 0;
    for (legate::PointInRectIterator<DIM> it{chars_shape}; it.valid(); ++it) {
      ASSERT_EQ(rw_chars[*it], i);
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
    auto rw_mask = null_mask.read_write_accessor<bool, 1>();

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

void test_fill_string_array_task(legate::LogicalArray& logical_array, bool nullable, bool unbound)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto context  = runtime->find_library(Config::LIBRARY_NAME);
  auto arr_type = legate::string_type();
  auto task     = runtime->create_task(context, FillStringTask::TASK_ID);
  auto part     = task.declare_partition();

  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));

  task = runtime->create_task(context, CheckStringTask::TASK_ID);
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(NullableFillStringArrayTest, BoundStringArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_array  = runtime->create_array({3}, legate::string_type(), nullable);
  test_fill_string_array_task(logical_array, nullable, false);
}

TEST_P(NullableFillStringArrayTest, UnboundStringArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_array  = runtime->create_array(legate::string_type(), 1, nullable);
  test_fill_string_array_task(logical_array, nullable, true);
}

}  // namespace physical_array_fill_string_test
