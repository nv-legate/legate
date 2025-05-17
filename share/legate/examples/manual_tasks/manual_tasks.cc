/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/utilities/detail/traced_exception.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>

// Legate automatic tasks partition data according to internal heuristics.
// However, there may be cases where the user wants to control how the
// data is partitioned. In such cases, manual tasks should be used.
// The following example demonstrates how to launch the
// InitTask on different subsets of an array by passing
// manually created partitionings to each task.

namespace {

constexpr std::int32_t EXPECTED_VALUE_A = 2000;
constexpr std::int32_t EXPECTED_VALUE_B = 2001;

class InitTask : public legate::LegateTask<InitTask> {
 public:
  //  This `TaskConfig` signature below does the following:
  //    - Assigns a local task ID of 1 ((should be unique within the library).
  //    - Specifies that the task takes exactly 1 output store
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(legate::TaskSignature{}.outputs(1));

  static void cpu_variant(legate::TaskContext context)
  {
    constexpr std::int32_t DIM = 2;
    // taks_index here corresponds to the task id in the Legate
    // index task launch. For this particular example, the number
    // of indices in the launch would be equal to the number of
    // tiles in created data  partitions
    auto&& task_index = context.get_task_index();
    auto output       = context.output(0).data();

    const auto shape = output.shape<DIM>();
    auto acc         = output.write_accessor<std::int32_t, DIM>();

    // Mark each element with task index
    const auto value = static_cast<std::int32_t>((task_index[0] * 1000) + task_index[1]);
    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      acc[*it] += value;
    }
  }
};

class VerificationTask : public legate::LegateTask<VerificationTask> {
 public:
  //  This `TaskConfig` signature below does the following:
  //    - Assigns a local task ID of 2 ((should be unique within the library).
  //    - Specifies that the task takes exactly 1 input store
  //    - Specifies that this task might be throwing exceptions
  //      this calling convention will be checked at runtime when
  //      the task is launched.

  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}
      .with_signature(legate::TaskSignature{}.inputs(1))
      .with_variant_options(legate::VariantOptions{}.with_may_throw_exception(true));

  static void cpu_variant(legate::TaskContext context)
  {
    constexpr std::int32_t DIM = 2;
    const auto input           = context.input(0).data();

    const auto shape = input.shape<DIM>();
    auto acc         = input.read_accessor<std::int32_t, DIM>();

    bool all_correct = true;
    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      const auto point      = *it;
      std::int32_t expected = 0;
      if (point[0] < 4 && point[1] > 3) {
        expected = 1;
      } else if (point[0] > 3 && point[1] < 4) {
        expected = EXPECTED_VALUE_A;
      } else if (point[0] > 3 && point[1] > 3) {
        expected = EXPECTED_VALUE_B;
      }
      const std::int32_t actual = acc[point];

      if (actual != expected) {
        std::cerr << "Mismatch at [" << point[0] << "," << point[1] << "]: expected " << expected
                  << ", got " << actual << "\n";
        all_correct = false;
      }
    }

    if (!all_correct) {
      // This example demonstrates exception handling in Legate.
      // To properly handle exceptions:
      // 1. Use the `legate::TaskException` API to throw exceptions.
      // 2. Notify Legate about potential exceptions by adding
      //    `.with_throws_exception(true)` when configuring TASK_CONFIG above.
      // This is necessary because Legate executes tasks asynchronously,
      // and proper exception handling ensures errors are propagated correctly.
      throw legate::detail::TracedException<std::runtime_error>("Verification failed");
    }
    std::cout << "Verification passed \n";
  }
};

void run_manual_task_example()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto library  = runtime->create_library("legate_manual");
  InitTask::register_variants(library);
  VerificationTask::register_variants(library);

  constexpr std::int32_t size      = 8;
  constexpr std::int32_t tile_size = 4;

  // Create and initialize store
  auto store = runtime->create_store(legate::Shape{size, size}, legate::int32());
  runtime->issue_fill(store, legate::Scalar{0});

  // the same "store" is passed to the InitTask 2 times using different
  // partitioning logic

  // Phase 1: we split original 8x8 array by rows with shape (4,8)
  {
    auto row_partition = store.partition_by_tiling({tile_size, size});

    auto task =
      runtime->create_task(library,
                           InitTask::TASK_CONFIG.task_id(),
                           legate::Domain{legate::Point<2>{0, 0}, legate::Point<2>{1, 0}});
    task.add_output(row_partition);
    runtime->submit(std::move(task));
  }

  // Phase 2: we splict original 8x8 array by tiles with shape (4x4)
  {
    auto grid_partition = store.partition_by_tiling({tile_size, tile_size});

    auto task = runtime->create_task(
      library, InitTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{2, 2});
    task.add_output(grid_partition);
    runtime->submit(std::move(task));
  }
  // Verify the results
  {
    auto verify_task = runtime->create_task(library, VerificationTask::TASK_CONFIG.task_id());
    verify_task.add_input(store);
    runtime->submit(std::move(verify_task));
  }

  // Demonstrates how to output a Legate array from the top-level context.
  //
  // WARNING: This approach requires the array to be inline-mapped, which has
  // significant performance implications:
  // 1. Implicit synchronization: Inline-mapping creates a blocking execution
  //    fence, forcing the runtime to wait for all pending tasks to complete
  //    before materializing the store.
  // 2. Full materialization: The entire store contents must be gathered into a
  //    single allocation. In parallel execution, this means each instance gets
  //    a complete copy of the data.
  //
  // Recommended alternative: Perform such operations within tasks because:
  // 1. Deferred execution: Tasks launch only when their inputs are ready,
  //    enabling better scheduling and pipelining.
  // 2. Partial access: Tasks operate on partitions of the store rather than
  //    requiring the full dataset, enabling distributed processing.
  // 3. No global synchronization: Tasks can proceed without waiting for
  //    all other work in the system to complete.
  {
    auto p_store = store.get_physical_store();
    auto acc     = p_store.read_accessor<std::int32_t, 2>();

    std::cout << "\nFinal array contents:\n";
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        std::cout << acc[{i, j}] << " ";
      }
      std::cout << "\n";
    }
  }
}

}  // namespace

int main()
{
  legate::start();
  run_manual_task_example();
  return legate::finish();
}
