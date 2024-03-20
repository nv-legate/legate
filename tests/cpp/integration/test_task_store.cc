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
#include <map>

namespace test_task_store {

// We really should NOT be silencing this warning in this file, since it uses an enormous
// amount of truly magic numbers (e.g. what is "index"), but I didn't write this and so I have
// no clue what to call the values.
//
// NOLINTBEGIN(readability-magic-numbers)

using Runtime = DefaultFixture;

enum class StoreType : std::uint8_t {
  NORMAL_STORE  = 0,
  UNBOUND_STORE = 1,
  SCALAR_STORE  = 2,
};

enum class TaskDataMode : std::uint8_t {
  INPUT     = 0,
  OUTPUT    = 1,
  REDUCTION = 2,
};

namespace {

constexpr const char library_name[] = "test_task_store";
constexpr std::int32_t INT_VALUE1   = 123;
constexpr std::int32_t INT_VALUE2   = 20;
constexpr std::int32_t SIMPLE_TASK  = 0;

[[nodiscard]] const std::map<TaskDataMode, std::int32_t>& auto_task_method_count()
{
  static const std::map<TaskDataMode, std::int32_t> map = {
    {TaskDataMode::INPUT, 2}, {TaskDataMode::OUTPUT, 2}, {TaskDataMode::REDUCTION, 4}};

  return map;
}

[[nodiscard]] const std::map<TaskDataMode, std::int32_t>& manual_task_method_count()
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

[[nodiscard]] const std::map<StoreType, std::int32_t>& create_store_count()
{
  static const std::map<StoreType, std::int32_t> map = {
    {StoreType::NORMAL_STORE, 2}, {StoreType::UNBOUND_STORE, 1}, {StoreType::SCALAR_STORE, 1}};

  return map;
}

}  // namespace

template <std::int32_t DIM>
class SimpleTask : public legate::LegateTask<SimpleTask<DIM>> {
 public:
  static constexpr std::int32_t TASK_ID = SIMPLE_TASK + DIM;

  static void cpu_variant(legate::TaskContext context)
  {
    auto dataMode  = static_cast<TaskDataMode>(context.scalar(0).value<std::uint32_t>());
    auto storeType = static_cast<StoreType>(context.scalar(1).value<std::uint32_t>());
    auto value     = context.scalar(2).value<std::int32_t>();

    legate::PhysicalStore store;
    switch (dataMode) {
      case TaskDataMode::INPUT: store = context.input(0).data(); break;
      case TaskDataMode::OUTPUT: store = context.output(0).data(); break;
      case TaskDataMode::REDUCTION: store = context.reduction(0).data(); break;
      default: break;
    }
    if (storeType == StoreType::UNBOUND_STORE) {
      store.bind_empty_data();
      if (dataMode == TaskDataMode::OUTPUT && context.output(0).nullable()) {
        context.output(0).null_mask().bind_empty_data();
      }
      return;
    }

    auto shape = store.shape<DIM>();
    if (shape.empty()) {
      return;
    }

    switch (dataMode) {
      case TaskDataMode::INPUT: {
        auto acc = store.read_accessor<int32_t, DIM>();
        for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
          EXPECT_EQ(acc[*it], value);
        }
      } break;
      case TaskDataMode::OUTPUT: {
        auto acc = store.write_accessor<int32_t, DIM>();
        for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
          acc[*it] = value;
        }
      } break;
      case TaskDataMode::REDUCTION: {
        auto acc = store.reduce_accessor<legate::SumReduction<std::int32_t>, true, DIM>();
        for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
          acc[*it].reduce(value);
        }
      } break;
      default: break;
    }
  }
};

struct VerifyOutputBody {
  template <std::int32_t DIM>
  void operator()(const legate::PhysicalStore& store, std::int32_t expected_value)
  {
    auto shape = store.shape<DIM>();
    if (shape.empty()) {
      return;
    }
    auto acc = store.read_accessor<int32_t, DIM>(shape);
    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      EXPECT_EQ(acc[*it], expected_value);
    }
  }
};

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
    case 2: array = runtime->create_array(shape, legate::int32(), false, true); break;
    case 3: {
      auto array1 = runtime->create_array(shape, legate::int32(), false, true);
      array       = runtime->create_array_like(array1);
    } break;
    case 4: array = runtime->create_array(shape, legate::int32(), true); break;
    case 5: {
      auto array1 = runtime->create_array(shape, legate::int32(), true);
      array       = runtime->create_array_like(array1);
    } break;
    case 6: array = runtime->create_array(shape, legate::int32(), true, true); break;
    case 7: {
      auto array1 = runtime->create_array(shape, legate::int32(), true, true);
      array       = runtime->create_array_like(array1);
    } break;
    default: break;
  }
  return array;
}

legate::LogicalArray create_unbound_array(std::uint32_t index, std::uint32_t ndim)
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::int32(), ndim, false);  // dummy creation
  switch (index) {
    case 0: array = runtime->create_array(legate::int32(), ndim, false); break;
    case 1: {
      auto array1 = runtime->create_array(legate::int32(), ndim, false);
      array       = runtime->create_array_like(array1);
    } break;
    case 2: array = runtime->create_array(legate::int32(), ndim, true); break;
    case 3: {
      auto array1 = runtime->create_array(legate::int32(), ndim, true);
      array       = runtime->create_array_like(array1);
    } break;
    default: break;
  }
  return array;
}

legate::LogicalStore create_normal_store(std::uint32_t index, const legate::Shape& shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store({1},
                                     legate::int32());  // dummy creation

  switch (index) {
    case 0: store = runtime->create_store(shape, legate::int32()); break;
    case 1: store = runtime->create_store(shape, legate::int32(), true); break;
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

void auto_task_normal_input(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + array.dim());

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::INPUT);
  constexpr auto store_type = static_cast<std::int32_t>(StoreType::NORMAL_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1 + index);

  runtime->issue_fill(array, legate::Scalar{std::int32_t{in_value1}});
  switch (index) {
    case 0: task.add_input(array); break;
    case 1: task.add_input(array, task.find_or_declare_partition(array)); break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));
}

void auto_task_normal_output(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + array.dim());

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::OUTPUT);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::NORMAL_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1 + index);

  switch (index) {
    case 0: task.add_output(array); break;
    case 1: task.add_output(array, task.find_or_declare_partition(array)); break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
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
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + array.dim());

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::REDUCTION);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::NORMAL_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1 + index);
  const auto in_value2      = static_cast<std::int32_t>(INT_VALUE2 + index);
  constexpr auto red_op     = legate::ReductionOpKind::ADD;

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
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
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
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + array.dim());

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
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + array.dim());

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::OUTPUT);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::UNBOUND_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1 + index);

  switch (index) {
    case 0: task.add_output(array); break;
    case 1: task.add_output(array, task.find_or_declare_partition(array)); break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  EXPECT_FALSE(array.unbound());
}

void auto_task_unbound_reduction(const legate::LogicalArray& array, std::uint32_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + array.dim());

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

void manual_task_normal_input(const legate::LogicalStore& store,
                              std::uint32_t index,
                              const legate::tuple<std::uint64_t>& launch_shape,
                              const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::INPUT);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::NORMAL_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1 + index);

  runtime->issue_fill(store, legate::Scalar{std::int32_t{in_value1}});
  switch (index) {
    case 0: task.add_input(store); break;
    case 1: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_input(part);
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));
}

void manual_task_normal_output(const legate::LogicalStore& store,
                               std::uint32_t index,
                               const legate::tuple<std::uint64_t>& launch_shape,
                               const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::OUTPUT);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::NORMAL_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1 + index);

  switch (index) {
    case 0: task.add_output(store); break;
    case 1: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_output(part);
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
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
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::REDUCTION);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::NORMAL_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1 + index);
  const auto in_value2      = static_cast<std::int32_t>(INT_VALUE2 + index);
  constexpr auto red_op     = legate::ReductionOpKind::ADD;
  auto red_part_flag        = false;

  runtime->issue_fill(store, legate::Scalar{std::int32_t{in_value1}});
  switch (index) {
    case 0: task.add_reduction(store, red_op); break;
    case 1: task.add_reduction(store, static_cast<std::int32_t>(red_op)); break;
    case 2: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_reduction(part, red_op);
      red_part_flag = true;
    } break;
    case 3: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_reduction(part, static_cast<std::int32_t>(red_op));
      red_part_flag = true;
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
  task.add_scalar_arg(legate::Scalar{in_value2});

  runtime->submit(std::move(task));

  auto multiple       = red_part_flag ? 1 : launch_shape.volume();
  auto expected_value = in_value1 + in_value2 * multiple;
  dim_dispatch(
    static_cast<int>(store.dim()), VerifyOutputBody{}, store.get_physical_store(), expected_value);
}

void manual_task_unbound_input(const legate::LogicalStore& store,
                               std::uint32_t index,
                               const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

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
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::OUTPUT);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::UNBOUND_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1 + index);

  switch (index) {
    case 0: task.add_output(store); break;
    default: break;
  }

  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));

  EXPECT_FALSE(store.unbound());
}

void manual_task_unbound_reduction(const legate::LogicalStore& store,
                                   std::uint32_t index,
                                   const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

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
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::INPUT);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::SCALAR_STORE);
  constexpr auto in_value1  = static_cast<std::int32_t>(INT_VALUE1);

  switch (index) {
    case 0: task.add_input(store); break;
    case 1: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_input(part);
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
  task.add_scalar_arg(legate::Scalar{in_value1});

  runtime->submit(std::move(task));
}

void manual_task_scalar_output(const legate::LogicalStore& store,
                               std::uint32_t index,
                               const legate::tuple<std::uint64_t>& launch_shape,
                               const std::vector<std::uint64_t>& tile_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::OUTPUT);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::SCALAR_STORE);
  constexpr auto in_value1  = static_cast<std::int32_t>(INT_VALUE1);

  switch (index) {
    case 0: task.add_output(store); break;
    case 1: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_output(part);
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
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
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, SIMPLE_TASK + store.dim(), launch_shape);

  constexpr auto mode       = static_cast<std::uint32_t>(TaskDataMode::REDUCTION);
  constexpr auto store_type = static_cast<std::uint32_t>(StoreType::SCALAR_STORE);
  const auto in_value1      = static_cast<std::int32_t>(INT_VALUE1);
  const auto in_value2      = static_cast<std::int32_t>(INT_VALUE2 + index);
  constexpr auto red_op     = legate::ReductionOpKind::ADD;

  switch (index) {
    case 0: task.add_reduction(store, red_op); break;
    case 1: task.add_reduction(store, static_cast<std::int32_t>(red_op)); break;
    case 2: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_reduction(part, red_op);
    } break;
    case 3: {
      auto part = store.partition_by_tiling(tile_shape);
      task.add_reduction(part, static_cast<std::int32_t>(red_op));
    } break;
    default: break;
  }
  task.add_scalar_arg(legate::Scalar{mode});
  task.add_scalar_arg(legate::Scalar{store_type});
  task.add_scalar_arg(legate::Scalar{in_value2});

  runtime->submit(std::move(task));

  auto expected_value = in_value1 + in_value2 * launch_shape.volume();
  dim_dispatch(
    static_cast<int>(store.dim()), VerifyOutputBody{}, store.get_physical_store(), expected_value);
}

void prepare()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared = true;

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);

  SimpleTask<1>::register_variants(context);
  SimpleTask<2>::register_variants(context);
  SimpleTask<3>::register_variants(context);
}

class AutoTaskNormal
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<std::int32_t, std::int32_t, legate::Shape>> {};
class AutoTaskNormalInput : public AutoTaskNormal {};
class AutoTaskNormalOutput : public AutoTaskNormal {};
class AutoTaskNormalReduction : public AutoTaskNormal {};

class AutoTaskUnbound
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<std::int32_t, std::int32_t, std::uint32_t>> {};
class AutoTaskUnboundInput : public AutoTaskUnbound {};
class AutoTaskUnboundOutput : public AutoTaskUnbound {};
class AutoTaskUnboundReduction : public AutoTaskUnbound {};

class ManualTaskNormal
  : public DefaultFixture,
    public ::testing::WithParamInterface<
      std::tuple<std::int32_t,
                 std::int32_t,
                 std::tuple<legate::Shape, legate::Shape, std::vector<std::uint64_t>>>> {};
class ManualTaskNormalInput : public ManualTaskNormal {};
class ManualTaskNormalOutput : public ManualTaskNormal {};
class ManualTaskNormalReduction : public ManualTaskNormal {};

class ManualTaskUnbound : public DefaultFixture,
                          public ::testing::WithParamInterface<
                            std::tuple<std::int32_t, std::tuple<std::uint32_t, legate::Shape>>> {};
class ManualTaskUnboundInput : public ManualTaskUnbound {};
class ManualTaskUnboundOutput : public ManualTaskUnbound {};
class ManualTaskUnboundReduction : public ManualTaskUnbound {};

class ManualTaskScalar
  : public DefaultFixture,
    public ::testing::WithParamInterface<
      std::tuple<std::int32_t, std::tuple<legate::Shape, std::vector<std::uint64_t>>>> {};
class ManualTaskScalarInput : public ManualTaskScalar {};
class ManualTaskScalarOutput : public ManualTaskScalar {};
class ManualTaskScalarReduction : public ManualTaskScalar {};

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
                                                       std::vector<std::uint64_t>({2, 2})))
                     // [failed-case-1], [failed-case-2]
                     // [error 39] LEGION ERROR: Invalid color space
                     // color for child 0 of logical partition (2,1,1)
                     // std::make_tuple(legate::Shape({3, 3}), legate::Shape({3, 3}),
                     // std::vector<std::size_t>({2, 2})), std::make_tuple(legate::Shape({3, 3}),
                     // legate::Shape({2, 2}), std::vector<std::size_t>({3, 3})))
                     ));

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
                                                       std::vector<std::uint64_t>({2, 2})))
                     // [failed-case-3], when create_store_index=1, fail,
                     // result: add[*it] = 0, expected: 124. 3 out of 9 results hit this error.
                     // std::make_tuple(legate::Shape({3, 3}), legate::Shape({2, 2, 2}),
                     // std::vector<std::size_t>({2, 2})))
                     ));

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
                     ::testing::Values(std::make_tuple(2, legate::Shape({3, 5})))

                       ));

INSTANTIATE_TEST_SUITE_P(
  TaskStoreTests,
  ManualTaskUnboundOutput,
  ::testing::Combine(
    ::testing::Range(0, 1), ::testing::Values(std::make_tuple(2, legate::Shape({3, 5})))
    // [failed-case-4], when create_store_index=1, fail
    // [error 609] LEGION ERROR: Output region 0 of task test_task_store::SimpleTask<2> (UID: 41) is
    // requested to have 2 dimensions, but the color space has 1 dimensions. Dimensionalities of
    // output regions must be the same as the color space's in global indexing mode.
    // std::make_tuple(2, legate::Shape({1})))
    ));

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
                      std::make_tuple(legate::Shape{2}, std::vector<std::uint64_t>({2})))
    // [failed-case-5],
    // Fill(in_value_1), Reduce(in_value_2)
    // if launch shape = {1}, result is in_value_2. Unexpected.
    // if launch shape = {2}, result is in_value_1 + launch_shape * in_value_2. Expected.
    // std::make_tuple(legate::Shape({1}), std::vector<std::size_t>({1})))
    ));

TEST_P(AutoTaskNormalInput, Basic)
{
  auto [index1, index2, shape] = GetParam();
  prepare();
  auto array = create_normal_array(index1, shape);
  auto_task_normal_input(array, index2);
}

TEST_P(AutoTaskNormalOutput, Basic)
{
  auto [index1, index2, shape] = GetParam();
  prepare();
  auto array = create_normal_array(index1, shape);
  auto_task_normal_output(array, index2);
}

TEST_P(AutoTaskNormalReduction, Basic)
{
  auto [index1, index2, shape] = GetParam();
  prepare();
  auto array = create_normal_array(index1, shape);
  auto_task_normal_reduction(array, index2);
}

TEST_P(AutoTaskUnboundInput, Basic)
{
  auto [index1, index2, ndim] = GetParam();
  prepare();
  auto array = create_unbound_array(index1, ndim);
  auto_task_unbound_input(array, index2);
}

TEST_P(AutoTaskUnboundOutput, Basic)
{
  auto [index1, index2, ndim] = GetParam();
  prepare();
  auto array = create_unbound_array(index1, ndim);
  auto_task_unbound_output(array, index2);
}

TEST_P(AutoTaskUnboundReduction, Basic)
{
  auto [index1, index2, ndim] = GetParam();
  prepare();
  auto array = create_unbound_array(index1, ndim);
  auto_task_unbound_reduction(array, index2);
}

TEST_P(ManualTaskNormalInput, Basic)
{
  auto [index1, index2, shapes]                = GetParam();
  auto [store_shape, launch_shape, tile_shape] = shapes;
  prepare();
  auto store = create_normal_store(index1, store_shape);
  manual_task_normal_input(store, index2, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskNormalOutput, Basic)
{
  auto [index1, index2, shapes]                = GetParam();
  auto [store_shape, launch_shape, tile_shape] = shapes;
  prepare();
  auto store = create_normal_store(index1, store_shape);
  manual_task_normal_output(store, index2, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskNormalReduction, Basic)
{
  auto [index1, index2, shapes]                = GetParam();
  auto [store_shape, launch_shape, tile_shape] = shapes;
  prepare();
  auto store = create_normal_store(index1, store_shape);
  manual_task_normal_reduction(store, index2, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskUnboundInput, Basic)
{
  auto [index, ndim_lanuch_shape] = GetParam();
  auto [ndim, launch_shape]       = ndim_lanuch_shape;
  prepare();
  auto store = create_unbound_store(ndim);
  manual_task_unbound_input(store, index, launch_shape.extents());
}

TEST_P(ManualTaskUnboundOutput, Basic)
{
  auto [index, ndim_lanuch_shape] = GetParam();
  auto [ndim, launch_shape]       = ndim_lanuch_shape;
  prepare();
  auto store = create_unbound_store(ndim);
  manual_task_unbound_output(store, index, launch_shape.extents());
}

TEST_P(ManualTaskUnboundReduction, Basic)
{
  auto [index, ndim_lanuch_shape] = GetParam();
  auto [ndim, launch_shape]       = ndim_lanuch_shape;
  prepare();
  auto store = create_unbound_store(ndim);
  manual_task_unbound_reduction(store, index, launch_shape.extents());
}

TEST_P(ManualTaskScalarInput, Basic)
{
  auto [index, shapes]            = GetParam();
  auto [launch_shape, tile_shape] = shapes;
  prepare();
  auto store = create_scalar_store();
  manual_task_scalar_input(store, index, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskScalarOutput, Basic)
{
  auto [index, shapes]            = GetParam();
  auto [launch_shape, tile_shape] = shapes;
  prepare();
  auto store = create_scalar_store();
  manual_task_scalar_output(store, index, launch_shape.extents(), tile_shape);
}

TEST_P(ManualTaskScalarReduction, Basic)
{
  const auto& [index, shapes]            = GetParam();
  const auto& [launch_shape, tile_shape] = shapes;
  prepare();
  auto store = create_scalar_store();
  manual_task_scalar_reduction(store, index, launch_shape.extents(), tile_shape);
}

TEST_F(Runtime, CreateTaskInvalid)
{
  prepare();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  EXPECT_THROW(static_cast<void>(runtime->create_task(context, -100)), std::out_of_range);
  EXPECT_THROW(
    static_cast<void>(runtime->create_task(context, -100, legate::tuple<std::uint64_t>{2, 3})),
    std::out_of_range);
  EXPECT_THROW(static_cast<void>(
                 runtime->create_task(context, SIMPLE_TASK, legate::tuple<std::uint64_t>{2, 3})),
               std::out_of_range);

  EXPECT_THROW(static_cast<void>(runtime->create_task(context,
                                                      SIMPLE_TASK + 1,
                                                      legate::tuple<std::uint64_t>{
                                                        0,
                                                      })),
               std::invalid_argument);

  EXPECT_THROW(
    static_cast<void>(runtime->create_task(context, SIMPLE_TASK, legate::tuple<std::uint64_t>{})),
    std::out_of_range);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace test_task_store
