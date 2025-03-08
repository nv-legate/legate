/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <legate/cuda/cuda.h>

#include <fmt/format.h>

#include <cuda_runtime.h>
#endif

namespace replicated_write_test {

#define CHECK_CUDA(...)                                                               \
  do {                                                                                \
    const cudaError_t __result__ = __VA_ARGS__;                                       \
    if (__result__ != cudaSuccess) {                                                  \
      throw std::runtime_error{                                                       \
        fmt::format("Internal CUDA failure with error {} ({}) in file {} at line {}", \
                    cudaGetErrorString(__result__),                                   \
                    cudaGetErrorName(__result__),                                     \
                    __FILE__,                                                         \
                    __LINE__)};                                                       \
    }                                                                                 \
  } while (0)

// NOLINTBEGIN(readability-magic-numbers)

namespace {

class WriterTask : public legate::LegateTask<WriterTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(false);

  static void cpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    for (auto& output : outputs) {
      auto shape = output.shape<2>();
      auto acc   = output.data().write_accessor<std::int64_t, 2>();
      for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
        acc[*it] = 42;
      }
    }
  }
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    for (auto& output : outputs) {
      auto shape  = output.shape<2>();
      auto acc    = output.data().write_accessor<std::int64_t, 2>();
      auto stream = context.get_task_stream();
      auto vals   = std::vector<std::int64_t>(shape.volume(), 42);
      auto* ptr   = acc.ptr(shape);

      CHECK_CUDA(cudaMemcpyAsync(
        ptr, vals.data(), sizeof(*ptr) * shape.volume(), cudaMemcpyHostToDevice, stream));
    }
  }
#endif
};

class TaskThatDoesNothing : public legate::LegateTask<TaskThatDoesNothing> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    // This task shouldn't make any updates, as we only want to exercise side effects from launching
    // the task.
  }
};

class CheckerTask : public legate::LegateTask<CheckerTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto&& inputs = context.inputs();
    for (auto&& input : inputs) {
      auto shape = input.shape<2>();
      if (shape.empty()) {
        return;
      }
      auto acc = input.data().read_accessor<std::int64_t, 2>();
      for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
        EXPECT_EQ(acc[*it], 42);
      }
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_replicated_write";
  static void registration_callback(legate::Library library)
  {
    WriterTask::register_variants(library);
    TaskThatDoesNothing::register_variants(library);
    CheckerTask::register_variants(library);
  }
};

class ReplicatedWrite : public RegisterOnceFixture<Config> {};

std::vector<legate::LogicalStore> perform_replicate_writes(
  legate::Library library,
  const legate::tuple<std::uint64_t>& extents,
  std::uint32_t num_out_stores,
  const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, WriterTask::TASK_CONFIG.task_id(), launch_shape);

  std::vector<legate::LogicalStore> out_stores;
  for (std::uint32_t idx = 0; idx < num_out_stores; ++idx) {
    auto& out_store = out_stores.emplace_back(
      runtime->create_store(extents, legate::int64(), true /*optimize_scalar*/));
    task.add_output(out_store);
  }
  runtime->submit(std::move(task));

  return out_stores;
}

void perform_reductions(legate::Library library,
                        const std::vector<legate::LogicalStore>& stores,
                        const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task =
    runtime->create_task(library, TaskThatDoesNothing::TASK_CONFIG.task_id(), launch_shape);
  for (auto&& store : stores) {
    task.add_reduction(store, legate::ReductionOpKind::ADD);
  }
  runtime->submit(std::move(task));
}

void perform_replicated_read_write(legate::Library library,
                                   const std::vector<legate::LogicalStore>& stores,
                                   const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task =
    runtime->create_task(library, TaskThatDoesNothing::TASK_CONFIG.task_id(), launch_shape);
  for (auto&& store : stores) {
    task.add_input(store);
    task.add_output(store);
  }
  runtime->submit(std::move(task));
}

void validate_output_inline(const legate::LogicalStore& store)
{
  auto p_store = store.get_physical_store();
  auto shape   = p_store.shape<2>();
  auto acc     = p_store.read_accessor<std::int64_t, 2>();
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    EXPECT_EQ(acc[*it], 42);
  }
}

void validate_outputs(const legate::Library& library,
                      const std::vector<legate::LogicalStore>& stores)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, CheckerTask::TASK_CONFIG.task_id());
  for (auto&& store : stores) {
    task.add_input(store);
  }
  runtime->submit(std::move(task));
}

void validate_outputs(const legate::Library& library,
                      const std::vector<legate::LogicalStore>& stores,
                      const legate::tuple<std::uint64_t>& launch_shape)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, CheckerTask::TASK_CONFIG.task_id(), launch_shape);
  for (auto&& store : stores) {
    task.add_input(store);
  }
  runtime->submit(std::move(task));
}

void test_auto_task(legate::Library library,
                    const legate::tuple<std::uint64_t>& extents,
                    std::uint32_t num_out_stores)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto in_store = runtime->create_store({8, 8}, legate::int64());

  runtime->issue_fill(in_store, legate::Scalar{int64_t{1}});

  auto task = runtime->create_task(library, WriterTask::TASK_CONFIG.task_id());

  task.add_input(in_store);

  std::vector<legate::LogicalStore> out_stores;
  for (std::uint32_t idx = 0; idx < num_out_stores; ++idx) {
    auto& out_store = out_stores.emplace_back(
      runtime->create_store(extents, legate::int64(), true /*optimize_scalar*/));
    auto part = task.add_output(out_store);
    task.add_constraint(legate::broadcast(part));
  }
  runtime->submit(std::move(task));

  validate_outputs(library, out_stores);

  for (auto&& out_store : out_stores) {
    validate_output_inline(out_store);
  }
}

void test_manual_task(legate::Library library,
                      const legate::tuple<std::uint64_t>& extents,
                      std::uint32_t num_out_stores)
{
  auto launch_shape = legate::tuple<std::uint64_t>{3, 3};
  auto out_stores   = perform_replicate_writes(library, extents, num_out_stores, launch_shape);

  validate_outputs(library, out_stores, launch_shape);

  // This reshapes future maps in the scalar stores
  validate_outputs(library, out_stores, legate::tuple<std::uint64_t>{9});

  // This extracts the first futures of the scalar stores
  validate_outputs(library, out_stores, legate::tuple<std::uint64_t>{4});

  for (auto&& out_store : out_stores) {
    validate_output_inline(out_store);
  }
}

void test_manual_task_with_replicated_read_write(legate::Library library,
                                                 const legate::tuple<std::uint64_t>& extents,
                                                 std::uint32_t num_out_stores)
{
  auto launch_shape = legate::tuple<std::uint64_t>{3, 3};
  auto out_stores   = perform_replicate_writes(library, extents, num_out_stores, launch_shape);

  perform_replicated_read_write(library, out_stores, launch_shape);

  validate_outputs(library, out_stores, launch_shape);

  // This reshapes future maps in the scalar stores
  validate_outputs(library, out_stores, legate::tuple<std::uint64_t>{9});

  // This extracts the first futures of the scalar stores
  validate_outputs(library, out_stores, legate::tuple<std::uint64_t>{4});

  for (auto&& out_store : out_stores) {
    validate_output_inline(out_store);
  }
}

void test_manual_task_with_reductions(legate::Library library,
                                      const legate::tuple<std::uint64_t>& extents,
                                      std::uint32_t num_out_stores)
{
  auto launch_shape = legate::tuple<std::uint64_t>{3, 3};
  auto out_stores   = perform_replicate_writes(library, extents, num_out_stores, launch_shape);

  perform_reductions(library, out_stores, launch_shape);

  // No need to test various launch shapes for the checker, as the scalar reduction will turn the
  // stores back to future-backed stores
  validate_outputs(library, out_stores, launch_shape);

  for (auto&& out_store : out_stores) {
    validate_output_inline(out_store);
  }
}

}  // namespace

TEST_F(ReplicatedWrite, AutoNonScalarSingle)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_auto_task(library, {2, 2}, 1);
}

TEST_F(ReplicatedWrite, AutoNonScalarMultiple)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_auto_task(library, {3, 3}, 3);
}

TEST_F(ReplicatedWrite, ManualNonScalarSingle)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_manual_task(library, {2, 2}, 1);
}

TEST_F(ReplicatedWrite, ManualNonScalarMultiple)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_manual_task(library, {3, 3}, 2);
}

TEST_F(ReplicatedWrite, AutoScalarSingle)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_auto_task(library, {1, 1}, 1);
}

TEST_F(ReplicatedWrite, AutoScalarMultiple)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_auto_task(library, {1, 1}, 5);
}

TEST_F(ReplicatedWrite, ManualScalarSingle)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_manual_task(library, {1, 1}, 1);
}

TEST_F(ReplicatedWrite, ManualScalarMultiple)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_manual_task(library, {1, 1}, 5);
}

TEST_F(ReplicatedWrite, ManualScalarMultipleWithReplicatedReadWrite)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_manual_task_with_replicated_read_write(library, {1, 1}, 3);
}

TEST_F(ReplicatedWrite, ManualScalarMultipleWithReductions)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  test_manual_task_with_reductions(library, {1, 1}, 3);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace replicated_write_test
