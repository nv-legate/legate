/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <integration/task_store/task_common.h>

namespace test_task_store {

namespace {

[[nodiscard]] const std::map<TaskDataMode, std::int32_t>& manual_task_method_count()
{
  static const std::map<TaskDataMode, std::int32_t> map = {
    {TaskDataMode::INPUT, 3}, {TaskDataMode::OUTPUT, 3}, {TaskDataMode::REDUCTION, 5}};
  return map;
}

[[nodiscard]] const std::map<StoreType, std::int32_t>& create_store_count()
{
  static const std::map<StoreType, std::int32_t> map = {
    {StoreType::NORMAL_STORE, 2}, {StoreType::UNBOUND_STORE, 1}, {StoreType::SCALAR_STORE, 1}};
  return map;
}

// Store creation functions
legate::LogicalStore create_normal_store(std::uint32_t index, const legate::Shape& shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store({1}, legate::int32());  // dummy creation

  switch (index) {
    case 0: store = runtime->create_store(shape, legate::int32()); break;
    case 1: store = runtime->create_store(shape, legate::int32(), true /*optimize_scalar*/); break;
    default: break;
  }
  return store;
}

legate::LogicalStore create_unbound_store(std::uint32_t ndim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int32(), ndim);
  return store;
}

legate::LogicalStore create_scalar_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{INT_VALUE1});
  return store;
}

legate::LogicalStore create_promote_store(const legate::Shape& shape,
                                          const std::vector<std::int32_t>& promote_args)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(shape, legate::int32());
  return store.promote(promote_args.at(0), promote_args.at(1));
}

// Manual task functions
void manual_task_normal_input(const legate::LogicalStore& store,
                              std::uint32_t index,
                              const legate::tuple<std::uint64_t>& launch_shape,
                              const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  const auto in_value1 = static_cast<std::int32_t>(INT_VALUE1 + index);

  runtime->issue_fill(store, legate::Scalar{std::int32_t{in_value1}});
  switch (index) {
    case 0: task.add_input(store); break;
    case 1: {
      if (launch_shape.size() == tile_shape.size()) {
        auto part = store.partition_by_tiling(tile_shape, launch_shape);
        task.add_input(part);
        break;
      }
    }
      [[fallthrough]];
    case 2: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_input(part);
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::INPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));
}

void manual_task_normal_output(const legate::LogicalStore& store,
                               std::uint32_t index,
                               const legate::tuple<std::uint64_t>& launch_shape,
                               const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  const auto in_value1 = static_cast<std::int32_t>(INT_VALUE1 + index);

  switch (index) {
    case 0: task.add_output(store); break;
    case 1: {
      if (launch_shape.size() == tile_shape.size()) {
        auto part = store.partition_by_tiling(tile_shape, launch_shape);
        task.add_output(part);
        break;
      }
    }
      [[fallthrough]];
    case 2: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_output(part);
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::OUTPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  auto expected_value = in_value1;
  dim_dispatch(
    static_cast<int>(store.dim()), VerifyOutputBody{}, store.get_physical_store(), expected_value);
}

void manual_task_normal_reduction(const legate::LogicalStore& store,
                                  std::uint32_t index,
                                  const legate::tuple<std::uint64_t>& launch_shape,
                                  const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  const auto in_value1  = static_cast<std::int32_t>(INT_VALUE1 + index);
  const auto in_value2  = static_cast<std::int32_t>(INT_VALUE2 + index);
  constexpr auto red_op = legate::ReductionOpKind::ADD;
  auto red_part_flag    = false;

  runtime->issue_fill(store, legate::Scalar{std::int32_t{in_value1}});
  switch (index) {
    case 0: task.add_reduction(store, red_op); break;
    case 1: task.add_reduction(store, static_cast<std::int32_t>(red_op)); break;
    case 2: {
      if (launch_shape.size() == tile_shape.size()) {
        auto part = store.partition_by_tiling(tile_shape);
        task.add_reduction(part, red_op);
        red_part_flag = true;
        break;
      }
    }
      [[fallthrough]];
    case 3: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_reduction(part, red_op);
      red_part_flag = true;
    } break;
    case 4: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_reduction(part, static_cast<std::int32_t>(red_op));
      red_part_flag = true;
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::REDUCTION});
  task.add_scalar_arg(legate::Scalar{StoreType::NORMAL_STORE});
  task.add_scalar_arg(legate::Scalar{in_value2});

  runtime->submit(std::move(task));

  auto multiple       = red_part_flag ? 1 : launch_shape.volume();
  auto expected_value = in_value1 + (in_value2 * multiple);
  dim_dispatch(
    static_cast<int>(store.dim()), VerifyOutputBody{}, store.get_physical_store(), expected_value);
}

void manual_task_unbound_input(const legate::LogicalStore& store,
                               std::uint32_t index,
                               const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  switch (index) {
    case 0: EXPECT_THROW(task.add_input(store), std::invalid_argument); break;
    default: break;
  }
}

void manual_task_unbound_output(const legate::LogicalStore& store,
                                std::uint32_t index,
                                const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  const auto in_value1 = static_cast<std::int32_t>(INT_VALUE1 + index);

  switch (index) {
    case 0: task.add_output(store); break;
    default: break;
  }

  task.add_scalar_arg(legate::Scalar{TaskDataMode::OUTPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::UNBOUND_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  EXPECT_FALSE(store.unbound());
}

void manual_task_unbound_reduction(const legate::LogicalStore& store,
                                   std::uint32_t index,
                                   const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  constexpr auto red_op = legate::ReductionOpKind::ADD;

  switch (index) {
    case 0: EXPECT_THROW(task.add_reduction(store, red_op), std::invalid_argument); break;
    case 1:
      EXPECT_THROW(task.add_reduction(store, static_cast<std::int32_t>(red_op)),
                   std::invalid_argument);
      break;
    default: break;
  }
}

void manual_task_scalar_input(const legate::LogicalStore& store,
                              std::uint32_t index,
                              const legate::tuple<std::uint64_t>& launch_shape,
                              const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  constexpr auto in_value1 = static_cast<std::int32_t>(INT_VALUE1);

  switch (index) {
    case 0: task.add_input(store); break;
    case 1: {
      if (launch_shape.size() == tile_shape.size()) {
        auto part = store.partition_by_tiling(tile_shape, launch_shape);
        task.add_input(part);
        break;
      }
    }
      [[fallthrough]];
    case 2: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_input(part);
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::INPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::SCALAR_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));
}

void manual_task_scalar_output(const legate::LogicalStore& store,
                               std::uint32_t index,
                               const legate::tuple<std::uint64_t>& launch_shape,
                               const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  constexpr auto in_value1 = static_cast<std::int32_t>(INT_VALUE1);

  switch (index) {
    case 0: task.add_output(store); break;
    case 1: {
      if (launch_shape.size() == tile_shape.size()) {
        auto part = store.partition_by_tiling(tile_shape, launch_shape);
        task.add_output(part);
        break;
      }
    }
      [[fallthrough]];
    case 2: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_output(part);
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::OUTPUT});
  task.add_scalar_arg(legate::Scalar{StoreType::SCALAR_STORE});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  auto expected_value = in_value1;
  dim_dispatch(
    static_cast<int>(store.dim()), VerifyOutputBody{}, store.get_physical_store(), expected_value);
}

void manual_task_scalar_reduction(const legate::LogicalStore& store,
                                  std::uint32_t index,
                                  const legate::tuple<std::uint64_t>& launch_shape,
                                  const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task =
    runtime->create_task(context,
                         legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + store.dim()},
                         launch_shape);

  const auto in_value1  = static_cast<std::int32_t>(INT_VALUE1);
  const auto in_value2  = static_cast<std::int32_t>(INT_VALUE2 + index);
  constexpr auto red_op = legate::ReductionOpKind::ADD;

  switch (index) {
    case 0: task.add_reduction(store, red_op); break;
    case 1: task.add_reduction(store, static_cast<std::int32_t>(red_op)); break;
    case 2: {
      if (launch_shape.size() == tile_shape.size()) {
        auto part = store.partition_by_tiling(tile_shape, launch_shape);
        task.add_reduction(part, red_op);
        break;
      }
    }
      [[fallthrough]];
    case 3: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_reduction(part, red_op);
    } break;
    case 4: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_reduction(part, static_cast<std::int32_t>(red_op));
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{TaskDataMode::REDUCTION});
  task.add_scalar_arg(legate::Scalar{StoreType::SCALAR_STORE});
  task.add_scalar_arg(legate::Scalar{in_value2});

  runtime->submit(std::move(task));

  auto expected_value = in_value1 + (in_value2 * launch_shape.volume());
  dim_dispatch(
    static_cast<int>(store.dim()), VerifyOutputBody{}, store.get_physical_store(), expected_value);
}

}  // namespace

// Test class definitions
class ManualTaskNormal
  : public TaskStoreTests,
    public ::testing::WithParamInterface<
      std::tuple<std::int32_t,
                 std::int32_t,
                 std::tuple<legate::Shape, legate::Shape, std::vector<std::uint64_t>>>> {};

class ManualTaskNormalInput : public ManualTaskNormal {};

class ManualTaskNormalOutput : public ManualTaskNormal {};

class ManualTaskNormalReduction : public ManualTaskNormal {};

class ManualTaskUnbound : public TaskStoreTests,
                          public ::testing::WithParamInterface<
                            std::tuple<std::int32_t, std::tuple<std::uint32_t, legate::Shape>>> {};

class ManualTaskUnboundInput : public ManualTaskUnbound {};

class ManualTaskUnboundOutput : public ManualTaskUnbound {};

class ManualTaskUnboundReduction : public ManualTaskUnbound {};

class ManualTaskScalar
  : public TaskStoreTests,
    public ::testing::WithParamInterface<
      std::tuple<std::int32_t, std::tuple<legate::Shape, std::vector<std::uint64_t>>>> {};

class ManualTaskScalarInput : public ManualTaskScalar {};

class ManualTaskScalarOutput : public ManualTaskScalar {};

class ManualTaskScalarReduction : public ManualTaskScalar {};

class ManualTaskPromote
  : public TaskStoreTests,
    public ::testing::WithParamInterface<std::tuple<std::int32_t,
                                                    std::tuple<legate::Shape,
                                                               legate::Shape,
                                                               std::vector<std::uint64_t>,
                                                               std::vector<std::int32_t>>>> {};

class ManualTaskPromoteInput : public ManualTaskPromote {};

class ManualTaskPromoteOutput : public ManualTaskPromote {};

class ManualTaskPromoteReduction : public ManualTaskPromote {};

// Test instantiations
INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskNormalInput,
  ::testing::Combine(::testing::Range(0, create_store_count().at(StoreType::NORMAL_STORE)),
                     ::testing::Range(0, manual_task_method_count().at(TaskDataMode::INPUT)),
                     ::testing::Values(std::make_tuple(legate::Shape({5, 5}),
                                                       legate::Shape({3, 3}),
                                                       std::vector<std::uint64_t>({2, 2})),
                                       std::make_tuple(legate::Shape({3, 3}),
                                                       legate::Shape({1}),
                                                       std::vector<std::uint64_t>({3, 3})),
                                       std::make_tuple(legate::Shape({3, 3}),
                                                       legate::Shape({2, 2}),
                                                       std::vector<std::uint64_t>({2, 2})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskNormalOutput,
  ::testing::Combine(::testing::Range(0, create_store_count().at(StoreType::NORMAL_STORE)),
                     ::testing::Range(0, manual_task_method_count().at(TaskDataMode::OUTPUT)),
                     ::testing::Values(std::make_tuple(legate::Shape({5, 5}),
                                                       legate::Shape({3, 3}),
                                                       std::vector<std::uint64_t>({2, 2})),
                                       std::make_tuple(legate::Shape({3, 3}),
                                                       legate::Shape({1}),
                                                       std::vector<std::uint64_t>({3, 3})),
                                       std::make_tuple(legate::Shape({3, 3}),
                                                       legate::Shape({2, 2}),
                                                       std::vector<std::uint64_t>({2, 2})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskNormalReduction,
  ::testing::Combine(::testing::Range(0, create_store_count().at(StoreType::NORMAL_STORE)),
                     ::testing::Range(0, manual_task_method_count().at(TaskDataMode::REDUCTION)),
                     ::testing::Values(std::make_tuple(legate::Shape({5, 5}),
                                                       legate::Shape({3, 3}),
                                                       std::vector<std::uint64_t>({2, 2})),
                                       std::make_tuple(legate::Shape({3, 3}),
                                                       legate::Shape({1}),
                                                       std::vector<std::uint64_t>({3, 3})),
                                       std::make_tuple(legate::Shape({3, 3}),
                                                       legate::Shape({2, 2}),
                                                       std::vector<std::uint64_t>({2, 2})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskUnboundInput,
  ::testing::Combine(::testing::Range(0, 1),
                     ::testing::Values(std::make_tuple(2, legate::Shape({3, 5})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskUnboundOutput,
  ::testing::Combine(::testing::Range(0, 1),
                     ::testing::Values(std::make_tuple(2, legate::Shape({3, 5})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskUnboundReduction,
  ::testing::Combine(::testing::Range(0, 2),
                     ::testing::Values(std::make_tuple(2, legate::Shape({3, 5})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskScalarInput,
  ::testing::Combine(
    ::testing::Range(0, manual_task_method_count().at(TaskDataMode::INPUT)),
    ::testing::Values(std::make_tuple(legate::Shape{{3, 5}}, std::vector<std::uint64_t>({2})),
                      std::make_tuple(legate::Shape{2}, std::vector<std::uint64_t>({2})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskScalarOutput,
  ::testing::Combine(
    ::testing::Range(0, manual_task_method_count().at(TaskDataMode::OUTPUT)),
    ::testing::Values(std::make_tuple(legate::Shape{{3, 5}}, std::vector<std::uint64_t>({2})),
                      std::make_tuple(legate::Shape{2}, std::vector<std::uint64_t>({2})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskScalarReduction,
  ::testing::Combine(
    ::testing::Range(0, manual_task_method_count().at(TaskDataMode::REDUCTION)),
    ::testing::Values(std::make_tuple(legate::Shape{{3, 5}}, std::vector<std::uint64_t>({2})),
                      std::make_tuple(legate::Shape{2}, std::vector<std::uint64_t>({2})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskPromoteInput,
  ::testing::Combine(::testing::Range(0, manual_task_method_count().at(TaskDataMode::INPUT)),
                     ::testing::Values(std::make_tuple(legate::Shape({3}),
                                                       legate::Shape({1}),
                                                       std::vector<std::uint64_t>({3, 1}),
                                                       std::vector<std::int32_t>({1, 1})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskPromoteOutput,
  ::testing::Combine(::testing::Range(0, manual_task_method_count().at(TaskDataMode::OUTPUT)),
                     ::testing::Values(std::make_tuple(legate::Shape({3}),
                                                       legate::Shape({1}),
                                                       std::vector<std::uint64_t>({3, 1}),
                                                       std::vector<std::int32_t>({1, 1})))));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskPromoteReduction,
  ::testing::Combine(::testing::Range(0, manual_task_method_count().at(TaskDataMode::REDUCTION)),
                     ::testing::Values(std::make_tuple(legate::Shape({3}),
                                                       legate::Shape({1}),
                                                       std::vector<std::uint64_t>({3, 1}),
                                                       std::vector<std::int32_t>({1, 1})))));

// Test implementations
TEST_P(ManualTaskNormalInput, Basic)
{
  auto [index1, index2, shapes]                = GetParam();
  auto [store_shape, launch_shape, tile_shape] = shapes;
  auto store                                   = create_normal_store(index1, store_shape);
  manual_task_normal_input(store, index2, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskNormalOutput, Basic)
{
  auto [index1, index2, shapes]                = GetParam();
  auto [store_shape, launch_shape, tile_shape] = shapes;
  auto store                                   = create_normal_store(index1, store_shape);
  manual_task_normal_output(store, index2, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskNormalReduction, Basic)
{
  auto [index1, index2, shapes]                = GetParam();
  auto [store_shape, launch_shape, tile_shape] = shapes;
  auto store                                   = create_normal_store(index1, store_shape);
  manual_task_normal_reduction(store, index2, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskUnboundInput, Basic)
{
  auto [index, ndim_lanuch_shape] = GetParam();
  auto [ndim, launch_shape]       = ndim_lanuch_shape;
  auto store                      = create_unbound_store(ndim);
  manual_task_unbound_input(store, index, launch_shape.extents());
}

TEST_P(ManualTaskUnboundOutput, Basic)
{
  auto [index, ndim_lanuch_shape] = GetParam();
  auto [ndim, launch_shape]       = ndim_lanuch_shape;
  auto store                      = create_unbound_store(ndim);
  manual_task_unbound_output(store, index, launch_shape.extents());
}

TEST_P(ManualTaskUnboundReduction, Basic)
{
  auto [index, ndim_lanuch_shape] = GetParam();
  auto [ndim, launch_shape]       = ndim_lanuch_shape;
  auto store                      = create_unbound_store(ndim);
  manual_task_unbound_reduction(store, index, launch_shape.extents());
}

TEST_P(ManualTaskScalarInput, Basic)
{
  auto [index, shapes]            = GetParam();
  auto [launch_shape, tile_shape] = shapes;
  auto store                      = create_scalar_store();
  manual_task_scalar_input(store, index, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskScalarOutput, Basic)
{
  auto [index, shapes]            = GetParam();
  auto [launch_shape, tile_shape] = shapes;
  auto store                      = create_scalar_store();
  manual_task_scalar_output(store, index, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskScalarReduction, Basic)
{
  const auto& [index, shapes]            = GetParam();
  const auto& [launch_shape, tile_shape] = shapes;
  auto store                             = create_scalar_store();
  manual_task_scalar_reduction(store, index, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskPromoteInput, Basic)
{
  auto [index, shapes]                                       = GetParam();
  auto [store_shape, launch_shape, tile_shape, promote_args] = shapes;
  auto store = create_promote_store(store_shape, promote_args);
  manual_task_normal_input(store, index, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskPromoteOutput, Basic)
{
  auto [index, shapes]                                       = GetParam();
  auto [store_shape, launch_shape, tile_shape, promote_args] = shapes;
  auto store = create_promote_store(store_shape, promote_args);
  manual_task_normal_output(store, index, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskPromoteReduction, Basic)
{
  auto [index, shapes]                                       = GetParam();
  auto [store_shape, launch_shape, tile_shape, promote_args] = shapes;
  auto store = create_promote_store(store_shape, promote_args);
  manual_task_normal_reduction(store, index, launch_shape.extents(), tile_shape);
}

TEST_F(TaskStoreTests, CreateTaskInvalid)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  EXPECT_THROW(static_cast<void>(runtime->create_task(context, legate::LocalTaskID{-100})),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(runtime->create_task(
                 context, legate::LocalTaskID{-100}, legate::tuple<std::uint64_t>{2, 3})),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(runtime->create_task(
                 context, legate::LocalTaskID{SIMPLE_TASK}, legate::tuple<std::uint64_t>{2, 3})),
               std::out_of_range);

  EXPECT_THROW(static_cast<void>(runtime->create_task(
                 context,
                 legate::LocalTaskID{static_cast<std::int64_t>(SIMPLE_TASK) + 1},
                 legate::tuple<std::uint64_t>{0})),
               std::invalid_argument);

  EXPECT_THROW(static_cast<void>(runtime->create_task(
                 context, legate::LocalTaskID{SIMPLE_TASK}, legate::tuple<std::uint64_t>{})),
               std::out_of_range);
}

}  // namespace test_task_store
