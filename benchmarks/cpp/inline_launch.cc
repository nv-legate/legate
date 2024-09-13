/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <benchmark/benchmark.h>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view LIBNAME = "bench";

class EmptyTask : public legate::LegateTask<EmptyTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static void cpu_variant(legate::TaskContext) {}
};

class TaskLaunchFixture : public benchmark::Fixture {
 public:
  static constexpr std::int32_t NUM_INPUTS_OUTPUTS = 10;

  void SetUp(benchmark::State& state) override
  {
    for (auto* dest : {&saved_inputs_, &saved_outputs_}) {
      dest->reserve(static_cast<std::size_t>(state.range(0)));
      for (std::int64_t i = 0; i < state.range(0); ++i) {
        dest->emplace_back(make_store_());
      }
    }
  }

  void TearDown(benchmark::State&) override
  {
    saved_inputs_.clear();
    saved_outputs_.clear();
  }

  [[nodiscard]] legate::LogicalStore add_input_store(std::size_t i)
  {
    return reuse_or_make_store_(i, saved_inputs_);
  }

  [[nodiscard]] legate::LogicalStore add_output_store(std::size_t i)
  {
    return reuse_or_make_store_(i, saved_outputs_);
  }

 private:
  [[nodiscard]] legate::LogicalStore reuse_or_make_store_(
    std::size_t i, const std::vector<legate::LogicalStore>& cache)
  {
    if (i < cache.size()) {
      return cache[i];
    }

    return make_store_();
  }

  [[nodiscard]] legate::LogicalStore make_store_() const
  {
    const auto runtime = legate::Runtime::get_runtime();
    auto store         = runtime->create_store(shape_, type_);

    LEGATE_CHECK(type_.code() == legate::Type::Code::INT32);
    runtime->issue_fill(store, legate::Scalar{std::int32_t{0}});
    return store;
  }

  [[nodiscard]] static legate::Shape make_shape_()
  {
    auto extents = legate::tuple<std::uint64_t>{};

    extents.reserve(LEGATE_MAX_DIM);
    for (std::uint64_t i = 1; i <= LEGATE_MAX_DIM; ++i) {
      extents.append_inplace(i);
    }
    return legate::Shape{extents};
  }

  legate::Shape shape_{make_shape_()};
  legate::Type type_{legate::int32()};
  std::vector<legate::LogicalStore> saved_inputs_{};
  std::vector<legate::LogicalStore> saved_outputs_{};
};

void benchmark_body(TaskLaunchFixture& fixt, benchmark::State& state)
{
  auto runtime = legate::Runtime::get_runtime();
  auto lib     = runtime->find_library(LIBNAME);

  for (auto _ : state) {
    state.PauseTiming();
    auto task = runtime->create_task(lib, EmptyTask::TASK_ID);
    for (std::size_t i = 0; i < TaskLaunchFixture::NUM_INPUTS_OUTPUTS; ++i) {
      task.add_input(fixt.add_input_store(i));
    }
    for (std::size_t i = 0; i < TaskLaunchFixture::NUM_INPUTS_OUTPUTS; ++i) {
      task.add_output(fixt.add_output_store(i));
    }
    state.ResumeTiming();

    runtime->submit(std::move(task));
    runtime->issue_execution_fence(true);
  }
}

BENCHMARK_DEFINE_F(TaskLaunchFixture, InlineTaskLaunch)(benchmark::State& state)
{
  benchmark_body(*this, state);
}

// NOLINTBEGIN(cert-err58-cpp)
BENCHMARK_REGISTER_F(TaskLaunchFixture, InlineTaskLaunch)
  ->Unit(benchmark::kMicrosecond)
  // Determines the number of reused inputs and outputs
  ->DenseRange(/* begin */ 0, /* end */ TaskLaunchFixture::NUM_INPUTS_OUTPUTS, /* step */ 2);
// NOLINTEND(cert-err58-cpp)

}  // namespace

int main(int argc, char** argv)
{
  if (const auto ret = legate::start(argc, argv); ret) {
    std::cerr << "Error starting legate, error code: " << ret << "\n";
    return ret;
  }
  EmptyTask::register_variants(legate::Runtime::get_runtime()->find_or_create_library(LIBNAME));

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return legate::finish();
}
