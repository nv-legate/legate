/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_array_fill_test {

namespace {

class FillBaseTask : public legate::LegateTask<FillBaseTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class CheckBaseTask : public legate::LegateTask<CheckBaseTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  static void cpu_variant(legate::TaskContext);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_fill_physical_array";

  static void registration_callback(legate::Library library)
  {
    FillBaseTask::register_variants(library);
    CheckBaseTask::register_variants(library);
  }
};

class FillPhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class NullableFillArrayTest : public FillPhysicalArrayUnit,
                              public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(FillPhysicalArrayUnit,
                         NullableFillArrayTest,
                         ::testing::Values(true, false));

void fill_bound_base_array(legate::PhysicalArray& array, bool nullable)
{
  auto store                        = array.data();
  static constexpr std::int32_t DIM = 2;
  auto w_store                      = store.write_accessor<std::int32_t, DIM>();
  auto store_shape                  = store.shape<DIM>();

  if (!store_shape.empty()) {
    auto i = 0;

    for (legate::PointInRectIterator<DIM> it{store_shape}; it.valid(); ++it) {
      w_store[*it] = i;
      i++;
    }
  }

  if (nullable) {
    auto null_mask  = array.null_mask();
    auto w_mask     = null_mask.write_accessor<bool, DIM>();
    auto mask_shape = null_mask.shape<DIM>();

    if (mask_shape.empty()) {
      return;
    }
    auto index = 0;

    for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
      w_mask[*it] = (index % 2 == 0);
      index++;
    }
  }
}

void fill_unbound_base_array(legate::PhysicalArray& array, bool nullable)
{
  auto store                        = array.data();
  static constexpr std::int32_t DIM = 2;

  ASSERT_NO_THROW(
    static_cast<void>(store.create_output_buffer<std::int32_t, 2>(legate::Point<DIM>{5}, true)));
  if (nullable) {
    auto null_mask = array.null_mask();
    ASSERT_NO_THROW(
      static_cast<void>(null_mask.create_output_buffer<bool, 2>(legate::Point<DIM>{5}, true)));
  }
}

void check_bound_base_array(legate::PhysicalArray& array, bool nullable)
{
  auto store                        = array.data();
  static constexpr std::int32_t DIM = 2;
  auto r_store                      = store.read_accessor<std::int32_t, DIM>();
  auto store_shape                  = store.shape<DIM>();

  if (!store_shape.empty()) {
    auto i = 0;

    for (legate::PointInRectIterator<DIM> it{store_shape}; it.valid(); ++it) {
      ASSERT_EQ(r_store[*it], i);
      i++;
    }
  }

  if (nullable) {
    auto null_mask  = array.null_mask();
    auto r_mask     = null_mask.read_accessor<bool, DIM>();
    auto mask_shape = null_mask.shape<DIM>();

    if (mask_shape.empty()) {
      return;
    }
    auto index = 0;

    for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
      ASSERT_EQ(r_mask[*it], (index % 2 == 0));
      index++;
    }
  }
}

void check_unbound_base_array(legate::PhysicalArray& array, bool nullable)
{
  auto store                        = array.data();
  static constexpr std::int32_t DIM = 2;
  auto rw_store                     = store.read_write_accessor<std::int32_t, DIM>();
  auto store_shape                  = store.shape<DIM>();

  if (!store_shape.empty()) {
    auto i = 2;

    for (legate::PointInRectIterator<DIM> it{store_shape}; it.valid(); ++it) {
      rw_store[*it] = i;
      i++;
    }
    i = 2;
    for (legate::PointInRectIterator<DIM> it{store_shape}; it.valid(); ++it) {
      ASSERT_EQ(rw_store[*it], i);
      i++;
    }
  }

  if (nullable) {
    auto null_mask  = array.null_mask();
    auto r_mask     = null_mask.read_write_accessor<bool, DIM>();
    auto mask_shape = null_mask.shape<DIM>();

    if (mask_shape.empty()) {
      return;
    }
    auto index = 0;

    for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
      r_mask[*it] = (index % 2 == 0);
      index++;
    }
    index = 0;
    for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
      ASSERT_EQ(r_mask[*it], (index % 2 == 0));
      index++;
    }
  }
}

/*static*/ void FillBaseTask::cpu_variant(legate::TaskContext context)
{
  auto array    = context.output(0);
  auto nullable = context.scalar(0).value<bool>();
  auto unbound  = context.scalar(1).value<bool>();

  if (unbound) {
    fill_unbound_base_array(array, nullable);
  } else {
    fill_bound_base_array(array, nullable);
  }
}

/*static*/ void CheckBaseTask::cpu_variant(legate::TaskContext context)
{
  auto array    = context.output(0);
  auto nullable = context.scalar(0).value<bool>();
  auto unbound  = context.scalar(1).value<bool>();

  if (unbound) {
    check_unbound_base_array(array, nullable);
  } else {
    check_bound_base_array(array, nullable);
  }
}

void test_fill_array_task(legate::LogicalArray& logical_array, bool nullable, bool unbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, FillBaseTask::TASK_ID);
  auto part    = task.declare_partition();

  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));

  task = runtime->create_task(context, CheckBaseTask::TASK_ID);
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(NullableFillArrayTest, BoundPrimitiveArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_array  = runtime->create_array({1, 4}, legate::int32(), nullable);
  test_fill_array_task(logical_array, nullable, false);
}

TEST_P(NullableFillArrayTest, UnboundPrimitiveArray)
{
  const auto nullable       = GetParam();
  auto runtime              = legate::Runtime::get_runtime();
  static constexpr auto DIM = 2;
  auto logical_array        = runtime->create_array(legate::int32(), DIM, nullable);
  test_fill_array_task(logical_array, nullable, true);
}

}  // namespace physical_array_fill_test
