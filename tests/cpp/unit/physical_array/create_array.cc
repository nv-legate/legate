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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace physical_array_create_test {

namespace {

class UnboundArrayTask : public legate::LegateTask<UnboundArrayTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_create_physical_array";

  static void registration_callback(legate::Library library)
  {
    UnboundArrayTask::register_variants(library);
  }
};

class CreatePhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class NullableCreateArrayTest : public CreatePhysicalArrayUnit,
                                public ::testing::WithParamInterface<bool> {};

class BoundPhysicalArrayTest
  : public CreatePhysicalArrayUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, bool, legate::Rect<2>>> {};

INSTANTIATE_TEST_SUITE_P(CreatePhysicalArrayUnit,
                         NullableCreateArrayTest,
                         ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(
  CreatePhysicalArrayUnit,
  BoundPhysicalArrayTest,
  ::testing::Values(std::make_tuple(legate::Shape{2, 4}, true, legate::Rect<2>({0, 0}, {1, 3})),
                    std::make_tuple(legate::Shape{2, 4}, false, legate::Rect<2>({0, 0}, {1, 3}))));

/*static*/ void UnboundArrayTask::cpu_variant(legate::TaskContext context)
{
  auto array                        = context.output(0);
  auto nullable                     = context.scalar(0).value<bool>();
  auto store                        = array.data();
  static constexpr std::int32_t DIM = 3;

  ASSERT_TRUE(store.is_unbound_store());
  ASSERT_NO_THROW(static_cast<void>(
    store.create_output_buffer<std::uint32_t, DIM>(legate::Point<DIM>(10), true)));

  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.dim(), DIM);
  ASSERT_EQ(array.type(), legate::uint32());
  ASSERT_FALSE(array.nested());
  ASSERT_THROW(static_cast<void>(array.shape<DIM>()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.domain()), std::invalid_argument);

  ASSERT_TRUE(store.is_unbound_store());
  ASSERT_FALSE(store.is_future());
  ASSERT_EQ(store.dim(), DIM);
  ASSERT_THROW(static_cast<void>(store.shape<DIM>()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(store.domain()), std::invalid_argument);
  ASSERT_EQ(store.type(), legate::uint32());

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_TRUE(null_mask.is_unbound_store());
    ASSERT_NO_THROW(
      static_cast<void>(null_mask.create_output_buffer<bool, DIM>(legate::Point<DIM>(10), true)));
    ASSERT_THROW(static_cast<void>(null_mask.shape<DIM>()), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(null_mask.domain()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.as_list_array()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.as_string_array()), std::invalid_argument);
}

}  // namespace

TEST_P(BoundPhysicalArrayTest, Create)
{
  const auto [shape, nullable, bound_rect] = GetParam();
  auto runtime                             = legate::Runtime::get_runtime();
  auto type                                = legate::int64();
  auto logical_array                       = runtime->create_array(shape, type, nullable);
  auto array                               = logical_array.get_physical_array();
  static constexpr std::int32_t DIM        = 2;

  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.dim(), DIM);
  ASSERT_EQ(array.type(), type);
  ASSERT_FALSE(array.nested());
  ASSERT_EQ(array.shape<DIM>(), bound_rect);
  ASSERT_EQ((array.domain().bounds<DIM, std::int64_t>()), bound_rect);

  auto store = array.data();

  ASSERT_FALSE(store.is_unbound_store());
  ASSERT_FALSE(store.is_future());
  ASSERT_EQ(store.dim(), DIM);
  ASSERT_EQ(store.shape<DIM>(), bound_rect);
  ASSERT_EQ(store.type(), type);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.shape<DIM>(), array.shape<DIM>());
    ASSERT_EQ(null_mask.domain(), array.domain());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
}

TEST_P(NullableCreateArrayTest, InvalidBoundArrayChild)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_array = runtime->create_array(legate::Shape{1, 3, 4}, legate::uint32(), GetParam());
  auto array         = logical_array.get_physical_array();

  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

TEST_P(NullableCreateArrayTest, InvalidCastBoundArray)
{
  auto runtime = legate::Runtime::get_runtime();
  auto logical_array =
    runtime->create_array(legate::Shape{1, 3, 4, 2}, legate::int32(), GetParam());
  auto array = logical_array.get_physical_array();

  ASSERT_THROW(static_cast<void>(array.as_list_array()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.as_string_array()), std::invalid_argument);
}

TEST_P(NullableCreateArrayTest, UnboundArray)
{
  const auto nullable               = GetParam();
  auto runtime                      = legate::Runtime::get_runtime();
  auto context                      = runtime->find_library(Config::LIBRARY_NAME);
  static constexpr std::int32_t DIM = 3;
  auto logical_array                = runtime->create_array(legate::uint32(), DIM, nullable);
  auto task                         = runtime->create_task(context, UnboundArrayTask::TASK_ID);
  auto part                         = task.declare_partition();

  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  runtime->submit(std::move(task));
  ASSERT_FALSE(logical_array.unbound());
}

}  // namespace physical_array_create_test
