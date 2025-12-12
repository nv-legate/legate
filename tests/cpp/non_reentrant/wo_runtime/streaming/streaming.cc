/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/operation/detail/task.h>
#include <legate/redop/redop.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/variant_info.h>
#include <legate/utilities/detail/formatters.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gtest/gtest.h>

#include <mutex>
#include <utilities/env.h>
#include <utilities/utilities.h>
#include <utility>
#include <vector>

namespace test_streaming {

namespace {

class GlobalTaskOrder {
 public:
  [[nodiscard]] std::pair<const std::vector<legate::LocalTaskID>&,
                          const std::vector<legate::DomainPoint>&>
  execution_order() const
  {
    return {task_ids_, domain_points_};
  }

  void append(legate::LocalTaskID task_id, const legate::DomainPoint& dp)
  {
    const auto _ = std::scoped_lock<std::mutex>{mut_};

    task_ids_.emplace_back(task_id);
    domain_points_.emplace_back(dp);
  }

  void clear()
  {
    const auto _ = std::scoped_lock<std::mutex>{mut_};

    task_ids_.clear();
    domain_points_.clear();
  }

 private:
  std::mutex mut_{};
  std::vector<legate::LocalTaskID> task_ids_{};
  std::vector<legate::DomainPoint> domain_points_{};
};

[[nodiscard]] GlobalTaskOrder& GLOBAL_EXEC_ORDER()
{
  static auto order = GlobalTaskOrder{};

  return order;
}

class InputTask : public legate::LegateTask<InputTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(
      legate::TaskSignature{}.inputs(2).outputs(1));

  static void cpu_variant(legate::TaskContext ctx)
  {
    GLOBAL_EXEC_ORDER().append(TASK_CONFIG.task_id(), ctx.get_task_index());
  }
};

class OutputTask : public legate::LegateTask<OutputTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(
      legate::TaskSignature{}.outputs(2).inputs(1));

  static void cpu_variant(legate::TaskContext ctx)
  {
    GLOBAL_EXEC_ORDER().append(TASK_CONFIG.task_id(), ctx.get_task_index());
  }
};

class NegateTask : public legate::LegateTask<NegateTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void cpu_variant(legate::TaskContext context)
  {
    constexpr auto DIM = 2;
    const auto arr     = context.output(0);
    const auto acc     = arr.data().read_write_accessor<std::int64_t, DIM>();

    for (legate::PointInRectIterator<DIM> it{arr.shape<DIM>()}; it.valid(); ++it) {
      acc[*it] *= -1;
    }
  }
};

class SumTask : public legate::LegateTask<SumTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{3}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext context)
  {
    const auto in      = context.input(0);
    const auto red     = context.reduction(0);
    const auto in_acc  = in.data().read_accessor<std::int64_t, 2>();
    const auto red_acc = red.data().reduce_accessor<legate::SumReduction<std::int64_t>, true, 1>();

    for (legate::PointInRectIterator<2> it{in.shape<2>()}; it.valid(); ++it) {
      red_acc.reduce(0, in_acc[*it]);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_streaming";

  static void registration_callback(legate::Library library)
  {
    InputTask::register_variants(library);
    OutputTask::register_variants(library);
    NegateTask::register_variants(library);
    SumTask::register_variants(library);
  }
};

[[nodiscard]] legate::LogicalStore make_store(const legate::Shape& shape)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto ret            = runtime->create_store(shape, legate::int64());

  runtime->issue_fill(ret, legate::Scalar{std::int64_t{0}});
  legate::detail::log_legate().print() << "Create store " << ret.to_string();
  return ret;
}

void launch_output_task(const legate::LogicalStore& store, const legate::LogicalStore& dep_store)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, OutputTask::TASK_CONFIG.task_id());

  task.add_output(store);
  task.add_input(dep_store);
  task.add_output(dep_store);
  runtime->submit(std::move(task));
}

void launch_input_task(const legate::LogicalStore& store, const legate::LogicalStore& dep_store)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, InputTask::TASK_CONFIG.task_id());

  task.add_input(store);
  task.add_input(dep_store);
  task.add_output(dep_store);
  runtime->submit(std::move(task));
}

void check_task_flip_flops()
{
  legate::Runtime::get_runtime()->issue_execution_fence(/* block */ true);

  auto&& [task_ids, domain_points] = GLOBAL_EXEC_ORDER().execution_order();

  // Should be a set of pairs, because each set of tasks are a pair of output->input
  ASSERT_FALSE(task_ids.empty());
  ASSERT_EQ(task_ids.size() % 2, 0);
  ASSERT_EQ(task_ids.size(), domain_points.size());
  // Normally tasks execute "horizontally", where all instances of a particular task are
  // executed before any dependent operations. But in streaming, the tasks are executed
  // "vertically", where for each submitted task, the same slice of leaf tasks are executed.
  //
  // Horizontal Execution (Batch mode). All instances of Task A complete before Task B starts.
  //
  // Time ->
  // +-------+ +-------+ +-------+     // Task A instances
  // | A[1]  | | A[2]  | | A[3]  |
  // +-------+ +-------+ +-------+
  //      v        v        v
  // +-------+ +-------+ +-------+     // Task B instances
  // | B[1]  | | B[2]  | | B[3]  |
  // +-------+ +-------+ +-------+
  //
  // Vertical Execution (Streaming mode). Each "slice" of the pipeline is completed before the
  // next one starts.
  //
  // Time ->
  // +-------+       +-------+       +-------+
  // | A[1]  | --->  | B[1]  | --->  | C[1]  |    // Slice 1
  // +-------+       +-------+       +-------+
  // +-------+       +-------+       +-------+
  // | A[2]  | --->  | B[2]  | --->  | C[2]  |    // Slice 2
  // +-------+       +-------+       +-------+
  // +-------+       +-------+       +-------+
  // | A[3]  | --->  | B[3]  | --->  | C[3]  |    // Slice 3
  // +-------+       +-------+       +-------+
  //
  // Crucially, even though we vertically *schedule* the tasks (in our mappers
  // `select_tasks_to_map()`), Legion is still at liberty to execute horizontally, so we cannot
  // just check we did A[1] -> B[1] -> C[1] -> A[2], etc.
  //
  // Instead, we should just check that we got some *some* interleaving, which should never
  // occur without vertical scheduling and pointwise analysis. In short, we are looking to see
  // that exactly:
  //
  // - OutputTask point(something)
  // - InputTask point(something)
  // - OutputTask point(something)
  //
  // happened at some point in the execution order. It is not sufficient to check that it
  // flipped from output to input (or vice versa) because that would also happen "normally".
  const auto flip_flop = {OutputTask::TASK_CONFIG.task_id(),
                          InputTask::TASK_CONFIG.task_id(),
                          OutputTask::TASK_CONFIG.task_id()};
  const auto it =
    std::search(task_ids.begin(), task_ids.end(), std::begin(flip_flop), std::end(flip_flop));

  ASSERT_NE(it, task_ids.end()) << "Failed to find vertical task interleaving in "
                                << fmt::format("{}", task_ids);
}

}  // namespace

class StreamingUnit : public RegisterOnceFixture<Config> {
 public:
  void SetUp() override
  {
    ASSERT_NO_THROW(legate::start());
    RegisterOnceFixture::SetUp();
  }

  void TearDown() override
  {
    RegisterOnceFixture::TearDown();
    ASSERT_EQ(legate::finish(), 0);
  }

 private:
  legate::test::Environment::TemporaryEnvVar legate_config_{
    "LEGATE_CONFIG",
    // We need to make sure Legion is constrained enough to want to vertically execute. Even
    // if we vertically map, Legion might still schedule the tasks horizontally. It will only
    // be forced to execute vertically if it thinks it's going to run out of memory.
    "--sysmem 1",
    /* overwrite */ true};
};

TEST_F(StreamingUnit, Basic)
{
  auto* const runtime = legate::Runtime::get_runtime();
  // Shape of the store is irrelevant, but we need it to be big enough to potentially bump up
  // against the upper sysmem limit so that Legion vertically schedules.
  const auto shape = legate::Shape{5, 5};

  runtime->issue_execution_fence(/* block */ true);
  GLOBAL_EXEC_ORDER().clear();
  {
    const auto _ =
      legate::Scope{legate::ParallelPolicy{}
                      .with_streaming(legate::StreamingMode::RELAXED)
                      .with_overdecompose_factor(static_cast<std::uint32_t>(shape.volume()))};

    // This is a dummy "dependency" store. It's sole purpose is to make sure the following 4
    // tasks execute in order. Each task takes this store as both an input and an output, so
    // this should ensure they execute as written.
    const auto dep_store = make_store(shape);

    {
      const auto store = make_store(shape);

      launch_output_task(store, dep_store);
      launch_input_task(store, dep_store);
    }
    {
      const auto store = make_store(shape);

      launch_output_task(store, dep_store);
      launch_input_task(store, dep_store);
    }
  }
  check_task_flip_flops();
}

namespace {

class MockOperation : public legate::detail::Operation {
 public:
  static constexpr auto UNIQUE_ID = 999999999999999999;

  MockOperation() : Operation{UNIQUE_ID} {}

  // We don't care what kind we are, just as long as we aren't a Task (which would be processed
  // by streaming code), or a discard/release-region-field
  Kind kind() const override { return Operation::Kind::TIMING; }

  // We set needs_flush() to false so that we enter (and sit in) the queue.
  bool needs_flush() const override { return false; }

  // We override strategy-less launch()
  bool needs_partitioning() const override { return false; }
};

class RecursiveFlush final : public MockOperation {
 public:
  void launch() override
  {
    // When we are launched, this will cause a recursive flush of the scheduling window.
    legate::detail::Runtime::get_runtime().flush_scheduling_window();
  }
};

}  // namespace

TEST_F(StreamingUnit, RecursiveFlush)
{
  auto* const runtime = legate::Runtime::get_runtime();
  // Shape of the store is irrelevant, but we need it to be big enough to potentially bump up
  // against the upper sysmem limit so that Legion vertically schedules.
  const auto shape = legate::Shape{5, 5};

  runtime->issue_execution_fence(/* block */ true);
  GLOBAL_EXEC_ORDER().clear();
  {
    const auto _ =
      legate::Scope{legate::ParallelPolicy{}
                      .with_streaming(legate::StreamingMode::RELAXED)
                      .with_overdecompose_factor(static_cast<std::uint32_t>(shape.volume()))};

    // This is a dummy "dependency" store. It's sole purpose is to make sure the following 4
    // tasks execute in order. Each task takes this store as both an input and an output, so
    // this should ensure they execute as written.
    const auto dep_store = make_store(shape);

    {
      const auto store = make_store(shape);

      launch_output_task(store, dep_store);
      launch_input_task(store, dep_store);
    }
    legate::detail::Runtime::get_runtime().submit(legate::make_internal_shared<RecursiveFlush>());
    {
      const auto store = make_store(shape);

      launch_output_task(store, dep_store);
      launch_input_task(store, dep_store);
    }
  }

  check_task_flip_flops();
}

namespace {

class RecursiveFlushAndAppend final : public MockOperation {
 public:
  explicit RecursiveFlushAndAppend(legate::LogicalStore store_) : store{std::move(store_)} {}

  void launch() override
  {
    // Before we flush, launch another pair of tasks to append to the scheduling queue. These
    // tasks should exist in their own streaming generation, while the others are left
    // alone. If:
    //
    // 1. The streaming run processing code does not leave the other tasks alone, the program
    //    will deadlock, because the mapper will endlessly wait for the full prior generation
    //    to arrive (which it might not, because it now has a new generation).
    // 2. The streaming run processing code does not assign these tasks to a new generation,
    launch_input_task(store, store);
    launch_output_task(store, store);
    legate::detail::Runtime::get_runtime().flush_scheduling_window();
  }

  legate::LogicalStore store;
};

}  // namespace

TEST_F(StreamingUnit, RecursiveFlushAndAppend)
{
  auto* const runtime = legate::Runtime::get_runtime();
  // Shape of the store is irrelevant, but we need it to be big enough to potentially bump up
  // against the upper sysmem limit so that Legion vertically schedules.
  const auto shape = legate::Shape{5, 5};

  runtime->issue_execution_fence(/* block */ true);
  GLOBAL_EXEC_ORDER().clear();
  {
    const auto _ =
      legate::Scope{legate::ParallelPolicy{}
                      .with_streaming(legate::StreamingMode::RELAXED)
                      .with_overdecompose_factor(static_cast<std::uint32_t>(shape.volume()))};

    // This is a dummy "dependency" store. It's sole purpose is to make sure the following 4
    // tasks execute in order. Each task takes this store as both an input and an output, so
    // this should ensure they execute as written.
    const auto dep_store = make_store(shape);

    {
      const auto store = make_store(shape);

      launch_output_task(store, dep_store);
      launch_input_task(store, dep_store);
    }
    legate::detail::Runtime::get_runtime().submit(
      legate::make_internal_shared<RecursiveFlushAndAppend>(dep_store));
    {
      const auto store = make_store(shape);

      launch_output_task(store, dep_store);
      launch_input_task(store, dep_store);
    }
  }

  check_task_flip_flops();
}

namespace {

constexpr std::uint32_t NUM_ITER = 4;
constexpr std::uint64_t EXT      = 10;
constexpr std::int64_t MAGIC     = 42;

void launch_negate_task(const legate::LogicalArray& array)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto library  = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, NegateTask::TASK_CONFIG.task_id());

  task.add_input(array);
  task.add_output(array);
  runtime->submit(std::move(task));
}

void launch_sum_task(const legate::LogicalArray& input, const legate::LogicalArray& sum)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto library  = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, SumTask::TASK_CONFIG.task_id());

  task.add_input(input);
  task.add_reduction(sum, legate::ReductionOpKind::ADD);
  runtime->submit(std::move(task));
}

void validate_pointwise(const legate::LogicalArray& array)
{
  constexpr auto DIM = 2;
  const auto p_array = array.get_physical_array();
  const auto acc     = p_array.data().read_accessor<std::int64_t, DIM>();

  for (legate::PointInRectIterator<DIM> it{p_array.shape<DIM>()}; it.valid(); ++it) {
    const auto expected = MAGIC * (NUM_ITER % 2 == 0 ? 1 : -1);

    ASSERT_EQ(acc[*it], expected);
  }
}

void validate_sum(const legate::LogicalArray& array)
{
  const auto p_array  = array.get_physical_array();
  const auto acc      = p_array.data().read_accessor<std::int64_t, 1>();
  const auto expected = EXT * EXT * MAGIC * (NUM_ITER % 2 == 0 ? 1 : -1);

  ASSERT_EQ(acc[0], expected);
}

}  // namespace

TEST_F(StreamingUnit, Pointwise)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto array    = runtime->create_array(legate::Shape{EXT, EXT}, legate::int64());

  runtime->issue_fill(array, legate::Scalar{MAGIC});

  {
    const auto scope = legate::Scope{legate::ParallelPolicy{}
                                       .with_streaming(legate::StreamingMode::RELAXED)
                                       .with_overdecompose_factor(2)};

    for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
      const auto nested = legate::Scope{"Task " + std::to_string(idx)};

      launch_negate_task(array);
    }
  }

  validate_pointwise(array);
}

TEST_F(StreamingUnit, SumAfterPointwise)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto input    = runtime->create_array(legate::Shape{EXT, EXT}, legate::int64());
  const auto output   = runtime->create_array(legate::Shape{1},
                                            legate::int64(),
                                            /*nullable=*/false,
                                            /*optimize_scalar=*/true);

  runtime->issue_fill(input, legate::Scalar{MAGIC});
  runtime->issue_fill(output, legate::Scalar{std::int64_t{0}});

  {
    const auto scope = legate::Scope{legate::ParallelPolicy{}
                                       .with_streaming(legate::StreamingMode::RELAXED)
                                       .with_overdecompose_factor(2)};

    for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
      const auto nested = legate::Scope{"Task " + std::to_string(idx)};

      launch_negate_task(input);
    }
    launch_sum_task(input, output);
  }

  validate_sum(output);
}

TEST_F(StreamingUnit, DISABLED_SumForTemporary)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto output   = runtime->create_array(legate::Shape{1},
                                            legate::int64(),
                                            /*nullable=*/false,
                                            /*optimize_scalar=*/true);

  runtime->issue_fill(output, legate::Scalar{std::int64_t{0}});

  {
    const auto scope = legate::Scope{legate::ParallelPolicy{}
                                       .with_streaming(legate::StreamingMode::RELAXED)
                                       .with_overdecompose_factor(2)};
    const auto input = runtime->create_array(legate::Shape{EXT, EXT}, legate::int64());

    runtime->issue_fill(input, legate::Scalar{MAGIC});

    for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
      const auto nested = legate::Scope{"Task " + std::to_string(idx)};

      launch_negate_task(input);
    }
    launch_sum_task(input, output);
  }

  validate_sum(output);
}

TEST_F(StreamingUnit, NestedPolicy)
{
  auto* const runtime = legate::Runtime::get_runtime();
  // Shape of the store is irrelevant, but we need it to be big enough to potentially bump up
  // against the upper sysmem limit so that Legion vertically schedules.
  const auto shape = legate::Shape{5, 5};

  runtime->issue_execution_fence(/* block */ true);
  GLOBAL_EXEC_ORDER().clear();
  {
    const auto _ =
      legate::Scope{legate::ParallelPolicy{}
                      .with_streaming(legate::StreamingMode::RELAXED)
                      .with_overdecompose_factor(static_cast<std::uint32_t>(shape.volume()))};

    // This is a dummy "dependency" store. It's sole purpose is to make sure the following 4
    // tasks execute in order. Each task takes this store as both an input and an output, so
    // this should ensure they execute as written.
    const auto dep_store = make_store(shape);

    {
      const auto store = make_store(shape);

      launch_output_task(store, dep_store);
      launch_input_task(store, dep_store);
    }
    {
      const auto s =
        legate::Scope{legate::ParallelPolicy{}
                        .with_streaming(legate::StreamingMode::RELAXED)
                        .with_overdecompose_factor(static_cast<std::uint32_t>(shape.volume()))};
      const auto store = make_store(shape);

      launch_output_task(store, dep_store);
      launch_input_task(store, dep_store);
    }
  }
  check_task_flip_flops();
}

}  // namespace test_streaming
