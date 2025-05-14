/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/experimental/trace.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace streaming_test {

namespace {

constexpr std::uint32_t NUM_ITER = 4;
constexpr std::uint64_t EXT      = 10;
constexpr std::int64_t MAGIC     = 42;

class NegateTask : public legate::LegateTask<NegateTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};
  static void cpu_variant(legate::TaskContext context)
  {
    auto arr = context.output(0);
    auto acc = arr.data().read_write_accessor<std::int64_t, 2>();
    for (legate::PointInRectIterator<2> it{arr.shape<2>()}; it.valid(); ++it) {
      acc[*it] *= -1;
    }
  }
};

class SumTask : public legate::LegateTask<SumTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));
  static void cpu_variant(legate::TaskContext context)
  {
    auto in      = context.input(0);
    auto red     = context.reduction(0);
    auto in_acc  = in.data().read_accessor<std::int64_t, 2>();
    auto red_acc = red.data().reduce_accessor<legate::SumReduction<std::int64_t>, true, 1>();
    for (legate::PointInRectIterator<2> it{in.shape<2>()}; it.valid(); ++it) {
      red_acc.reduce(0, in_acc[*it]);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "streaming_test";
  static void registration_callback(legate::Library library)
  {
    NegateTask::register_variants(library);
    SumTask::register_variants(library);
  }
};

class Streaming : public RegisterOnceFixture<Config> {};

void launch_negate_task(const legate::LogicalArray& array)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, NegateTask::TASK_CONFIG.task_id());

  task.add_input(array);
  task.add_output(array);
  runtime->submit(std::move(task));
}

void launch_sum_task(const legate::LogicalArray& input, const legate::LogicalArray& sum)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, SumTask::TASK_CONFIG.task_id());

  task.add_input(input);
  task.add_reduction(sum, legate::ReductionOpKind::ADD);
  runtime->submit(std::move(task));
}

void validate_pointwise(const legate::LogicalArray& array)
{
  auto p_array = array.get_physical_array();
  auto acc     = p_array.data().read_accessor<std::int64_t, 2>();

  for (legate::PointInRectIterator<2> it{p_array.shape<2>()}; it.valid(); ++it) {
    auto expected = MAGIC * (NUM_ITER % 2 == 0 ? 1 : -1);
    ASSERT_EQ(acc[*it], expected);
  }
}

void validate_sum(const legate::LogicalArray& array)
{
  auto p_array = array.get_physical_array();
  auto acc     = p_array.data().read_accessor<std::int64_t, 1>();

  auto expected = EXT * EXT * MAGIC * (NUM_ITER % 2 == 0 ? 1 : -1);
  ASSERT_EQ(acc[0], expected);
}

}  // namespace

TEST_F(Streaming, Pointwise)
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::Shape{EXT, EXT}, legate::int64());
  runtime->issue_fill(array, legate::Scalar{MAGIC});

  {
    const legate::Scope scope{
      legate::ParallelPolicy{}.with_streaming(true).with_overdecompose_factor(2)};

    for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
      const legate::Scope nested{"Task " + std::to_string(idx)};
      launch_negate_task(array);
    }
  }

  validate_pointwise(array);
}

TEST_F(Streaming, SumAfterPointwise)
{
  auto runtime = legate::Runtime::get_runtime();
  auto input   = runtime->create_array(legate::Shape{EXT, EXT}, legate::int64());
  auto output  = runtime->create_array(
    legate::Shape{1}, legate::int64(), false /*nullable*/, true /*optimize_scalar*/);
  runtime->issue_fill(input, legate::Scalar{MAGIC});
  runtime->issue_fill(output, legate::Scalar{std::int64_t{0}});

  {
    const legate::Scope scope{
      legate::ParallelPolicy{}.with_streaming(true).with_overdecompose_factor(2)};

    for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
      const legate::Scope nested{"Task " + std::to_string(idx)};
      launch_negate_task(input);
    }
    launch_sum_task(input, output);
  }

  validate_sum(output);
}

TEST_F(Streaming, DISABLED_SumForTemporary)
{
  auto runtime = legate::Runtime::get_runtime();
  auto output  = runtime->create_array(
    legate::Shape{1}, legate::int64(), false /*nullable*/, true /*optimize_scalar*/);
  runtime->issue_fill(output, legate::Scalar{std::int64_t{0}});

  {
    const legate::Scope scope{
      legate::ParallelPolicy{}.with_streaming(true).with_overdecompose_factor(2)};

    auto input = runtime->create_array(legate::Shape{EXT, EXT}, legate::int64());
    runtime->issue_fill(input, legate::Scalar{MAGIC});

    for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
      const legate::Scope nested{"Task " + std::to_string(idx)};
      launch_negate_task(input);
    }
    launch_sum_task(input, output);
  }

  validate_sum(output);
}

}  // namespace streaming_test
