/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <repartition.h>

#include <legate.h>

#include <legate/timing/timing.h>

#include <realm/cmdline.h>

#include <cstdint>
#include <iostream>
#include <string_view>

struct Config {
  std::uint64_t repeat{5};
  std::uint64_t M{2048};
  std::uint64_t N{2048};
};

class Timer {
 public:
  void start()
  {
    legate::Runtime::get_runtime()->issue_execution_fence();
    chrono_start_ = std::chrono::steady_clock::now();
  }

  [[nodiscard]] double stop()
  {
    legate::Runtime::get_runtime()->issue_execution_fence();
    const auto end_time = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - chrono_start_).count();
  }

 private:
  std::optional<legate::timing::Time> start_time_{};
  std::chrono::steady_clock::time_point chrono_start_{};
};

void InitTask::cpu_variant(legate::TaskContext context) { (void)context; }

void RepartitionTask::cpu_variant(legate::TaskContext context) { (void)context; }

[[nodiscard]] double run_example(legate::Library library, std::uint64_t M, std::uint64_t N)
{
  Timer timer{};
  auto* const runtime = legate::Runtime::get_runtime();

  auto shape   = legate::Shape{M, N};
  auto x_store = runtime->create_store(shape, legate::int32());

  auto init_x_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id());
  init_x_task.add_output(x_store);
  // init_x_task.add_constraint(legate::broadcast(part, {0}));
  init_x_task.add_scalar_arg(legate::Scalar{42});
  runtime->submit(std::move(init_x_task));

  timer.start();
  init_x_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id());
  init_x_task.add_output(x_store);
  runtime->submit(std::move(init_x_task));

  auto repartition_task = runtime->create_task(library, RepartitionTask::TASK_CONFIG.task_id());
  auto part_x           = repartition_task.add_input(x_store);
  repartition_task.add_constraint(legate::broadcast(part_x, {0}));
  runtime->submit(std::move(repartition_task));

  return timer.stop();
}

int main(int argc, char** argv)
{
  int num_runs      = 5;
  double total_time = 0.0;

  Config config{};
  Realm::CommandLineParser cp;
  cp.add_option_int("--repeat", config.repeat)
    .add_option_int("--M", config.M)
    .add_option_int("--N", config.N)
    .parse_command_line(argc, argv);

  legate::start();
  auto* const runtime = legate::Runtime::get_runtime();

  if (runtime->node_id() == 0) {
    std::cout << "========================================================================="
              << std::endl;
    std::cout << "Running Repartition with M: " << config.M << " N: " << config.N << " for "
              << config.repeat << " runs " << std::endl;
    std::cout << "========================================================================="
              << std::endl;
  }

  auto library = runtime->create_library("repartition");

  InitTask::register_variants(library);
  RepartitionTask::register_variants(library);

  auto warmup = run_example(library, config.M, config.N);
  if (runtime->node_id() == 0) {
    std::cout << "Warmup time: " << warmup << " ms" << std::endl;
  }

  for (std::uint64_t i = 0; i < config.repeat; ++i) {
    auto duration = run_example(library, config.M, config.N);
    total_time += duration;
    if (runtime->node_id() == 0) {
      std::cout << "Run " << i + 1 << " time: " << duration << " ms" << std::endl;
    }
  }

  if (num_runs > 1) {
    if (runtime->node_id() == 0) {
      std::cout << "Average time: " << total_time / num_runs << " ms" << std::endl;
    }
  }

  return legate::finish();
}
