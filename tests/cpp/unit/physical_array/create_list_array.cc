/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

namespace physical_array_create_list_test {

namespace {

class ListArrayTask : public legate::LegateTask<ListArrayTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_create_list_physical_array";

  static void registration_callback(legate::Library library)
  {
    ListArrayTask::register_variants(library);
  }
};

class CreateListPhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class NullableCreateListArrayTest : public CreateListPhysicalArrayUnit,
                                    public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(CreateListPhysicalArrayUnit,
                         NullableCreateListArrayTest,
                         ::testing::Values(true, false));

void test_array_data(legate::PhysicalStore& store,
                     bool is_unbound,
                     legate::Type::Code code,
                     std::int32_t dim)
{
  ASSERT_EQ(store.is_unbound_store(), is_unbound);
  ASSERT_EQ(store.dim(), dim);
  ASSERT_EQ(store.type().code(), code);
  if (is_unbound) {
    ASSERT_THROW(static_cast<void>(store.shape<1>()), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(store.domain()), std::invalid_argument);
  }
}

/*static*/ void ListArrayTask::cpu_variant(legate::TaskContext context)
{
  auto array                                = context.output(0);
  auto nullable                             = context.scalar(0).value<bool>();
  auto unbound                              = context.scalar(1).value<bool>();
  static constexpr std::int32_t DIM         = 1;
  auto list_array                           = array.as_list_array();
  auto descriptor_store                     = list_array.descriptor().data();
  auto vardata_store                        = list_array.vardata().data();
  static constexpr std::int64_t SHAPE_BOUND = 100;
  auto buffer =
    vardata_store.create_output_buffer<std::int64_t, DIM>(legate::Point<1>{SHAPE_BOUND}, true);

  if (unbound) {
    ASSERT_NO_THROW(descriptor_store.bind_empty_data());
  }

  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.dim(), DIM);
  ASSERT_EQ(array.type().code(), legate::list_type(legate::int64()).code());
  ASSERT_TRUE(array.nested());
  if (unbound) {
    ASSERT_THROW(static_cast<void>(array.shape<DIM>()), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(array.domain()), std::invalid_argument);
  }

  if (nullable) {
    auto null_mask = array.null_mask();

    if (null_mask.is_unbound_store()) {
      ASSERT_NO_THROW(null_mask.bind_empty_data());
      ASSERT_THROW(static_cast<void>(null_mask.shape<DIM>()), std::invalid_argument);
      ASSERT_THROW(static_cast<void>(null_mask.domain()), std::invalid_argument);
    }
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  test_array_data(descriptor_store, unbound, legate::Type::Code::STRUCT, DIM);
  test_array_data(vardata_store, true, legate::Type::Code::INT64, DIM);

  auto desc = array.child(0).data();
  auto var  = array.child(1).data();

  test_array_data(desc, unbound, legate::Type::Code::STRUCT, DIM);
  test_array_data(var, true, legate::Type::Code::INT64, DIM);

  ASSERT_THROW(static_cast<void>(array.child(2)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(array.child(-1)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(array.as_string_array()), std::invalid_argument);
}

void test_create_list_array_task(legate::LogicalArray& logical_array, bool nullable, bool unbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, ListArrayTask::TASK_ID);
  auto part    = task.declare_partition();

  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(NullableCreateListArrayTest, BoundListArray)
{
  const auto nullable                       = GetParam();
  auto runtime                              = legate::Runtime::get_runtime();
  auto arr_type                             = legate::list_type(legate::int64()).as_list_type();
  static constexpr std::int32_t SHAPE_BOUND = 6;
  auto logical_array = runtime->create_array({SHAPE_BOUND}, arr_type, nullable);

  test_create_list_array_task(logical_array, nullable, false);
}

TEST_P(NullableCreateListArrayTest, UnboundListArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto logical_array  = runtime->create_array(arr_type, 1, nullable);

  test_create_list_array_task(logical_array, nullable, true);
}

}  // namespace physical_array_create_list_test
