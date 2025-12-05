/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <integration/copy_util.inl>
#include <utilities/utilities.h>

// extern so that compilers don't also complain that function is unused!
extern void silence_function_warnings_unused() { static_cast<void>(::fill_indirect); }

namespace is_partitioned {

namespace {

constexpr std::uint64_t EXTENT    = 42;
constexpr std::uint64_t NUM_TASKS = 2;

class Tester : public legate::LegateTask<Tester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    const auto output1 = context.output(0).data();
    const auto output2 = context.output(1).data();
    const auto output3 = context.output(2).data();

    if (!context.is_single_task()) {
      EXPECT_TRUE(output1.is_partitioned());
      EXPECT_TRUE(output2.is_partitioned());
    }
    EXPECT_FALSE(output3.is_partitioned());

    output2.bind_empty_data();
  }
};

class CopyTask : public legate::LegateTask<CopyTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{10}};

  class CopyTaskBody {
   public:
    template <int DIM>
    void operator()(legate::TaskContext& context)
    {
      if (context.num_inputs() > 0) {
        // Copy from input to output
        const auto input  = context.input(0).data();
        const auto output = context.output(0).data();
        const auto shape  = input.shape<DIM>();

        if (shape.empty()) {
          return;
        }

        auto input_acc  = input.read_accessor<std::int64_t, DIM>(shape);
        auto output_acc = output.write_accessor<std::int64_t, DIM>(shape);

        for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
          output_acc[*it] = input_acc[*it];
        }
      } else {
        // Just write zeros to output (for initialization)
        const auto output = context.output(0).data();
        const auto shape  = output.shape<DIM>();

        if (shape.empty()) {
          return;
        }

        auto output_acc = output.write_accessor<std::int64_t, DIM>(shape);
        for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
          output_acc[*it] = 0;
        }
      }
    }
  };

  static void cpu_variant(legate::TaskContext context)
  {
    auto dim =
      context.num_inputs() > 0 ? context.input(0).data().dim() : context.output(0).data().dim();
    legate::dim_dispatch(dim, CopyTaskBody{}, context);
  }
};

class SumTask : public legate::LegateTask<SumTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{11}};

  static void cpu_variant(legate::TaskContext context)
  {
    const auto input1 = context.input(0).data();
    const auto input2 = context.input(1).data();
    const auto output = context.output(0).data();
    const auto shape  = input1.shape<1>();

    if (shape.empty()) {
      return;
    }

    auto input1_acc = input1.read_accessor<std::int64_t, 1>(shape);
    auto input2_acc = input2.read_accessor<std::int64_t, 1>(shape);
    auto output_acc = output.write_accessor<std::int64_t, 1>(shape);

    for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
      output_acc[*it] = input1_acc[*it] + input2_acc[*it];
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_is_partitioned";

  static void registration_callback(legate::Library library)
  {
    Tester::register_variants(library);
    // Register FillTask from copy_util.inl for 1D arrays
    FillTask<1>::register_variants(library);
    FillTask<2>::register_variants(library);
    CopyTask::register_variants(library);
    SumTask::register_variants(library);
  }
};

class IsPartitioned : public RegisterOnceFixture<Config> {};

class PartitionedStore : public RegisterOnceFixture<Config> {};

}  // namespace

TEST_F(IsPartitioned, Auto)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto store1 = runtime->create_store(legate::Shape{EXTENT}, legate::int64());
  auto store2 = runtime->create_store(legate::int64());
  auto store3 = runtime->create_store(legate::Shape{EXTENT}, legate::int64());

  auto task = runtime->create_task(library, Tester::TASK_CONFIG.task_id());
  task.add_output(store1);
  task.add_output(store2);
  auto part = task.add_output(store3);
  task.add_constraint(legate::broadcast(part));
  runtime->submit(std::move(task));
}

TEST_F(IsPartitioned, Manual)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto store1 = runtime->create_store(legate::Shape{EXTENT}, legate::int64());
  auto store2 = runtime->create_store(legate::int64());
  auto store3 = runtime->create_store(legate::Shape{EXTENT}, legate::int64());

  auto task = runtime->create_task(library, Tester::TASK_CONFIG.task_id(), {NUM_TASKS});
  task.add_output(store1.partition_by_tiling({EXTENT / NUM_TASKS}));
  task.add_output(store2);
  task.add_output(store3);
  runtime->submit(std::move(task));
}

TEST_F(PartitionedStore, StoreGetPartition)
{
  auto runtime                       = legate::Runtime::get_runtime();
  auto library                       = runtime->find_library(Config::LIBRARY_NAME);
  constexpr std::uint64_t total_size = 50000 * 32;
  const auto shape                   = legate::Shape{total_size};

  auto in_store = runtime->create_store(shape, legate::int64());

  fill_input(library, in_store, legate::Scalar{std::int64_t{1}});

  auto out_store = runtime->create_store(shape, legate::int64());
  auto copy_task = runtime->create_task(library, CopyTask::TASK_CONFIG.task_id());

  copy_task.add_input(in_store);
  copy_task.add_output(out_store);
  runtime->submit(std::move(copy_task));
  runtime->issue_execution_fence(true);

  auto second_input_store = runtime->create_store(shape, legate::int64());

  fill_input(library, second_input_store, legate::Scalar{std::int64_t{2}});

  std::vector<std::uint64_t> color_shape = {2, 1};

  auto out_partition_optional = out_store.get_partition();
  if (out_partition_optional.has_value()) {
    const auto& out_store_partition = out_partition_optional.value();
    auto out_color_shape            = out_store_partition.color_shape();

    ASSERT_EQ(out_color_shape.size(), 1);
    ASSERT_GT(out_color_shape[0], 0);

    color_shape = {out_color_shape.data()[0]};

    auto partition =
      second_input_store.partition_by_tiling({total_size / out_color_shape[0]}, out_color_shape);

    ASSERT_THAT(partition.color_shape(), testing::ContainerEq(out_color_shape));
  }

  auto manual_task = runtime->create_task(library, SumTask::TASK_CONFIG.task_id(), color_shape);
  auto sum_output_store = runtime->create_store(shape, legate::int64());

  manual_task.add_input(second_input_store);
  manual_task.add_input(out_store);
  manual_task.add_output(sum_output_store);
  runtime->submit(std::move(manual_task));
  runtime->issue_execution_fence(true);
}

TEST_F(PartitionedStore, PartitionPlacementInfo)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  constexpr std::uint64_t dim_x = 1000;
  constexpr std::uint64_t dim_y = 1280;
  const auto shape_2d           = legate::Shape{dim_x, dim_y};

  auto in_store_2d                  = runtime->create_store(shape_2d, legate::int64());
  auto out_store_2d                 = runtime->create_store(shape_2d, legate::int64());
  constexpr std::int64_t fill_value = 42;

  fill_input(library, in_store_2d, legate::Scalar{fill_value});

  auto copy_task_2d = runtime->create_task(library, CopyTask::TASK_CONFIG.task_id());
  copy_task_2d.add_input(in_store_2d);
  copy_task_2d.add_output(out_store_2d);
  runtime->submit(std::move(copy_task_2d));
  runtime->issue_execution_fence(true);

  auto partition_2d_optional = out_store_2d.get_partition();

  if (!partition_2d_optional.has_value()) {
    GTEST_SKIP() << "Partition not found, the task did not partition the store";
  }

  const auto& partition_2d   = partition_2d_optional.value();
  auto partition_color_shape = partition_2d.color_shape();

  ASSERT_GT(partition_color_shape.size(), 0) << "Partition should have non-zero dimensions";

  auto device_mapping_2d = partition_2d.get_placement_info();

  ASSERT_GT(device_mapping_2d.placements().size(), 0);

  std::uint64_t total_colors = 1;

  for (auto dim_size : partition_color_shape) {
    total_colors *= dim_size;
  }

  auto num_nodes             = runtime->node_count();
  auto expected_min_per_node = total_colors / num_nodes;

  EXPECT_GE(device_mapping_2d.placements().size(), expected_min_per_node)
    << "Node should have at least " << expected_min_per_node << " colors";
  EXPECT_GT(device_mapping_2d.placements().size(), 0);

  for (const auto& mapping : device_mapping_2d.placements()) {
    EXPECT_EQ(mapping.partition_color().size(), partition_color_shape.size())
      << "Partition color dimensionality mismatch";

    for (std::size_t dim = 0; dim < mapping.partition_color().size(); ++dim) {
      EXPECT_LT(mapping.partition_color()[dim], partition_color_shape[dim])
        << "Color coordinate " << dim << " out of range";
    }
  }
}

}  // namespace is_partitioned
