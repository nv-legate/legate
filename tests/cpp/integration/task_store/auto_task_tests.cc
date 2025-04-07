/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <integration/task_store/task_common.h>

namespace test_task_store {

namespace {

[[nodiscard]] const std::map<TaskDataMode, std::int32_t>& auto_task_method_count()
{
  static const std::map<TaskDataMode, std::int32_t> map = {
    {TaskDataMode::INPUT, 2}, {TaskDataMode::OUTPUT, 2}, {TaskDataMode::REDUCTION, 4}};
  return map;
}

[[nodiscard]] const std::map<StoreType, std::int32_t>& create_array_count()
{
  static const std::map<StoreType, std::int32_t> map = {{StoreType::NORMAL_STORE, 8},
                                                        {StoreType::UNBOUND_STORE, 4}};
  return map;
}

// NOLINTBEGIN(readability-magic-numbers)
legate::LogicalArray create_normal_array(std::uint32_t index, const legate::Shape& shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(shape, legate::int32());  // dummy creation

  switch (index) {
    case 0: array = runtime->create_array(shape, legate::int32()); break;
    case 1: {
      auto array1 = runtime->create_array(shape, legate::int32());
      array       = runtime->create_array_like(array1);
    } break;
    case 2:
      array =
        runtime->create_array(shape, legate::int32(), false /*nullable*/, true /*optimize_scalar*/);
      break;
    case 3: {
      auto array1 =
        runtime->create_array(shape, legate::int32(), false /*nullable*/, true /*optimize_scalar*/);
      array = runtime->create_array_like(array1);
    } break;
    case 4: array = runtime->create_array(shape, legate::int32(), true /*nullable*/); break;
    case 5: {
      auto array1 = runtime->create_array(shape, legate::int32(), true /*nullable*/);
      array       = runtime->create_array_like(array1);
    } break;
    case 6:
      array =
        runtime->create_array(shape, legate::int32(), true /*nullable*/, true /*optimize_scalar*/);
      break;
    case 7: {
      auto array1 =
        runtime->create_array(shape, legate::int32(), true /*nullable*/, true /*optimize_scalar*/);
      array = runtime->create_array_like(array1);
    } break;
    default: break;
  }
  return array;
}
// NOLINTEND(readability-magic-numbers)

legate::LogicalArray create_unbound_array(std::uint32_t index, std::uint32_t ndim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto array = runtime->create_array(legate::int32(), ndim, false /*nullable*/);  // dummy creation
  switch (index) {
    case 0: array = runtime->create_array(legate::int32(), ndim, false /*nullable*/); break;
    case 1: {
      auto array1 = runtime->create_array(legate::int32(), ndim, false /*nullable*/);
      array       = runtime->create_array_like(array1);
    } break;
    case 2: array = runtime->create_array(legate::int32(), ndim, true /*nullable*/); break;
    case 3: {
      auto array1 = runtime->create_array(legate::int32(), ndim, true /*nullable*/);
      array       = runtime->create_array_like(array1);
    } break;
    default: break;
  }
  return array;
}

void auto_task_normal_input(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + array.dim()});

  const auto in_value1 = static_cast<std::int32_t>(INT_VALUE1 + index);

  runtime->issue_fill(array, legate::Scalar{std::int32_t{in_value1}});
  switch (index) {
    case 0: task.add_input(array); break;
    case 1: task.add_input(array, task.find_or_declare_partition(array)); break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::INPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));
}

void auto_task_normal_output(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + array.dim()});

  const auto in_value1 = static_cast<std::int32_t>(INT_VALUE1 + index);

  switch (index) {
    case 0: task.add_output(array); break;
    case 1: task.add_output(array, task.find_or_declare_partition(array)); break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::OUTPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  auto expected_value = in_value1;
  dim_dispatch(static_cast<int>(array.dim()),
               VerifyOutputBody{},
               array.data().get_physical_store(),
               expected_value);
}

void auto_task_normal_reduction(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + array.dim()});

  const auto in_value1  = static_cast<std::int32_t>(INT_VALUE1 + index);
  const auto in_value2  = static_cast<std::int32_t>(INT_VALUE2 + index);
  constexpr auto red_op = legate::ReductionOpKind::ADD;

  runtime->issue_fill(array, legate::Scalar{std::int32_t{in_value1}});
  switch (index) {
    case 0: task.add_reduction(array, red_op); break;
    case 1: task.add_reduction(array, static_cast<std::int32_t>(red_op)); break;
    case 2: task.add_reduction(array, red_op, task.find_or_declare_partition(array)); break;
    case 3:
      task.add_reduction(
        array, static_cast<std::int32_t>(red_op), task.find_or_declare_partition(array));
      break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::REDUCTION});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value2});

  runtime->submit(std::move(task));

  auto expected_value = in_value1 + in_value2;
  dim_dispatch(static_cast<int>(array.dim()),
               VerifyOutputBody{},
               array.data().get_physical_store(),
               expected_value);
}

void auto_task_unbound_input(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + array.dim()});

  switch (index) {
    case 0: EXPECT_THROW(task.add_input(array), std::invalid_argument); break;
    case 1:
      EXPECT_THROW(task.add_input(array, task.find_or_declare_partition(array)),
                   std::invalid_argument);
      break;
    default: break;
  }
}

void auto_task_unbound_output(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + array.dim()});

  const auto in_value1 = static_cast<std::int32_t>(INT_VALUE1 + index);

  switch (index) {
    case 0: task.add_output(array); break;
    case 1: task.add_output(array, task.find_or_declare_partition(array)); break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::OUTPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::UNBOUND_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  EXPECT_FALSE(array.unbound());
}

void auto_task_unbound_reduction(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + array.dim()});

  constexpr auto red_op = legate::ReductionOpKind::ADD;

  switch (index) {
    case 0: EXPECT_THROW(task.add_reduction(array, red_op), std::invalid_argument); break;
    case 1:
      EXPECT_THROW(task.add_reduction(array, static_cast<std::int32_t>(red_op)),
                   std::invalid_argument);
      break;
    case 2:
      EXPECT_THROW(task.add_reduction(array, red_op, task.find_or_declare_partition(array)),
                   std::invalid_argument);
      break;
    case 3:
      EXPECT_THROW(
        task.add_reduction(
          array, static_cast<std::int32_t>(red_op), task.find_or_declare_partition(array)),
        std::invalid_argument);
      break;
    default: break;
  }
}

}  // namespace

// Test class definitions
class AutoTaskNormal
  : public TaskStoreTests,
    public ::testing::WithParamInterface<std::tuple<std::int32_t, std::int32_t, legate::Shape>> {};
class AutoTaskNormalInput : public AutoTaskNormal {};
class AutoTaskNormalOutput : public AutoTaskNormal {};
class AutoTaskNormalReduction : public AutoTaskNormal {};

class AutoTaskUnbound
  : public TaskStoreTests,
    public ::testing::WithParamInterface<std::tuple<std::int32_t, std::int32_t, std::uint32_t>> {};
class AutoTaskUnboundInput : public AutoTaskUnbound {};
class AutoTaskUnboundOutput : public AutoTaskUnbound {};
class AutoTaskUnboundReduction : public AutoTaskUnbound {};

// Test instantiations
INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  AutoTaskNormalInput,
  ::testing::Combine(::testing::Range(0, create_array_count().at(StoreType::NORMAL_STORE)),
                     ::testing::Range(0, auto_task_method_count().at(TaskDataMode::INPUT)),
                     ::testing::Values(legate::Shape({3, 3, 2}))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  AutoTaskNormalOutput,
  ::testing::Combine(::testing::Range(0, create_array_count().at(StoreType::NORMAL_STORE)),
                     ::testing::Range(0, auto_task_method_count().at(TaskDataMode::OUTPUT)),
                     ::testing::Values(legate::Shape({3, 3, 2}))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  AutoTaskNormalReduction,
  ::testing::Combine(::testing::Range(0, create_array_count().at(StoreType::NORMAL_STORE)),
                     ::testing::Range(0, auto_task_method_count().at(TaskDataMode::REDUCTION)),
                     ::testing::Values(legate::Shape({3, 3, 2}))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  AutoTaskUnboundInput,
  ::testing::Combine(::testing::Range(0, create_array_count().at(StoreType::UNBOUND_STORE)),
                     ::testing::Range(0, auto_task_method_count().at(TaskDataMode::INPUT)),
                     ::testing::Values(3, 1)));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  AutoTaskUnboundOutput,
  ::testing::Combine(::testing::Range(0, create_array_count().at(StoreType::UNBOUND_STORE)),
                     ::testing::Range(0, auto_task_method_count().at(TaskDataMode::OUTPUT)),
                     ::testing::Values(3, 1)));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  AutoTaskUnboundReduction,
  ::testing::Combine(::testing::Range(0, create_array_count().at(StoreType::UNBOUND_STORE)),
                     ::testing::Range(0, auto_task_method_count().at(TaskDataMode::REDUCTION)),
                     ::testing::Values(3, 1)));

// Test implementations
TEST_P(AutoTaskNormalInput, Basic)
{
  auto [index1, index2, shape] = GetParam();
  auto array                   = create_normal_array(index1, shape);
  auto_task_normal_input(array, index2);
}

TEST_P(AutoTaskNormalOutput, Basic)
{
  auto [index1, index2, shape] = GetParam();
  auto array                   = create_normal_array(index1, shape);
  auto_task_normal_output(array, index2);
}

TEST_P(AutoTaskNormalReduction, Basic)
{
  auto [index1, index2, shape] = GetParam();
  auto array                   = create_normal_array(index1, shape);
  auto_task_normal_reduction(array, index2);
}

TEST_P(AutoTaskUnboundInput, Basic)
{
  auto [index1, index2, ndim] = GetParam();
  auto array                  = create_unbound_array(index1, ndim);
  auto_task_unbound_input(array, index2);
}

TEST_P(AutoTaskUnboundOutput, Basic)
{
  auto [index1, index2, ndim] = GetParam();
  auto array                  = create_unbound_array(index1, ndim);
  auto_task_unbound_output(array, index2);
}

TEST_P(AutoTaskUnboundReduction, Basic)
{
  auto [index1, index2, ndim] = GetParam();
  auto array                  = create_unbound_array(index1, ndim);
  auto_task_unbound_reduction(array, index2);
}

}  // namespace test_task_store
