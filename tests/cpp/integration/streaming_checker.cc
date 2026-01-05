/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/detail/streaming/analysis.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace streaming_checker {

using ChildStore = DefaultFixture;

namespace {

using val_ty              = std::uint32_t;
constexpr std::size_t DIM = 2;

constexpr val_ty INIT_VAL = 3;

constexpr std::size_t EXTENT = 1024;

enum TaskIDs : std::uint8_t {
  INIT,
  SUM,
  CHECK,
};

class InitTask : public legate::LegateTask<InitTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{INIT}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto store_a = context.output(0).data();
    auto store_b = context.output(1).data();
    auto acc_a   = store_a.write_accessor<val_ty, DIM>();
    auto acc_b   = store_b.write_accessor<val_ty, DIM>();

    const auto shape = store_a.shape<DIM>();

    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      acc_a[*it] = INIT_VAL;
      acc_b[*it] = INIT_VAL;
    }
  }
};

class SumTask : public legate::LegateTask<SumTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{SUM}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto store_a = context.input(0).data();
    auto store_b = context.input(1).data();
    auto acc_a   = store_a.write_accessor<val_ty, DIM>();
    auto acc_b   = store_b.read_accessor<val_ty, DIM>();

    const auto shape = store_a.shape<DIM>();

    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      acc_a[*it] += acc_b[*it];
    }
  }
};

class CheckTask : public legate::LegateTask<CheckTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CHECK}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto store_a = context.input(0).data();
    auto store_b = context.input(1).data();
    auto acc_a   = store_a.read_accessor<val_ty, DIM>();
    auto acc_b   = store_b.read_accessor<val_ty, DIM>();

    const auto shape = store_a.shape<DIM>();

    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      ASSERT_EQ(acc_a[*it], INIT_VAL + INIT_VAL);
      ASSERT_EQ(acc_b[*it], INIT_VAL);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "streaming_checker";

  static void registration_callback(legate::Library library)
  {
    InitTask::register_variants(library);
    SumTask::register_variants(library);
    CheckTask::register_variants(library);
  }
};

std::pair<legate::LogicalStore, legate::LogicalStore> create_stores()
{
  auto runtime = legate::Runtime::get_runtime();
  auto store_a = runtime->create_store(legate::Shape{EXTENT, EXTENT}, legate::uint32());
  auto store_b = runtime->create_store(legate::Shape{EXTENT, EXTENT}, legate::uint32());

  return std::make_pair(store_a, store_b);
}

const legate::Scope* create_streaming_scope(legate::StreamingMode mode)
{
  auto pp = legate::ParallelPolicy{}.with_streaming(mode).with_overdecompose_factor(4);
  auto* s = new legate::Scope{};

  s->set_parallel_policy(pp);
  return s;
}

void launch_init_task(const legate::LogicalStore& store_a, const legate::LogicalStore& store_b)
{
  auto runtime   = legate::Runtime::get_runtime();
  auto library   = runtime->find_library(Config::LIBRARY_NAME);
  auto init_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id());

  init_task.add_output(store_a);
  init_task.add_output(store_b);

  runtime->submit(std::move(init_task));
}

void launch_sum_task(const legate::LogicalStore& store_a, const legate::LogicalStore& store_b)
{
  auto runtime  = legate::Runtime::get_runtime();
  auto library  = runtime->find_library(Config::LIBRARY_NAME);
  auto sum_task = runtime->create_task(library, SumTask::TASK_CONFIG.task_id());

  sum_task.add_output(store_a);
  sum_task.add_input(store_a);
  sum_task.add_input(store_b);

  runtime->submit(std::move(sum_task));
}

void launch_check_task(const legate::LogicalStore& store_a, const legate::LogicalStore& store_b)
{
  auto runtime    = legate::Runtime::get_runtime();
  auto library    = runtime->find_library(Config::LIBRARY_NAME);
  auto check_task = runtime->create_task(library, CheckTask::TASK_CONFIG.task_id());

  check_task.add_input(store_a);
  check_task.add_input(store_b);

  runtime->submit(std::move(check_task));
}

void launch_manual_init_task(const legate::LogicalStore& store_a,
                             const legate::LogicalStore& store_b)
{
  // launch a manual task with a different domain.
  auto runtime   = legate::Runtime::get_runtime();
  auto library   = runtime->find_library(Config::LIBRARY_NAME);
  auto init_task = runtime->create_task(library, InitTask::TASK_CONFIG.task_id(), {1, 2, 3, 4});
  init_task.add_output(store_a);
  init_task.add_output(store_b);
  runtime->submit(std::move(init_task));
}

}  // namespace

class StreamingChecker : public RegisterOnceFixture<Config>,
                         public ::testing::WithParamInterface<legate::StreamingMode> {};

INSTANTIATE_TEST_SUITE_P(StreamingCheckerSuite,
                         StreamingChecker,
                         ::testing::ValuesIn({legate::StreamingMode::OFF,
                                              legate::StreamingMode::RELAXED,
                                              legate::StreamingMode::STRICT}));

TEST_P(StreamingChecker, EmptyStreamingScope)
{
  const auto streaming_mode = GetParam();
  const auto* scope         = create_streaming_scope(streaming_mode);
  ASSERT_NO_THROW(delete scope);
}

TEST_P(StreamingChecker, SimpleScopeAutoTask)
{
  auto [store_a, store_b]   = create_stores();
  const auto streaming_mode = GetParam();
  const auto* scope         = create_streaming_scope(streaming_mode);

  launch_init_task(store_a, store_b);
  launch_sum_task(store_a, store_b);
  launch_check_task(store_a, store_b);

  // deleting the scope causes task scheduling window to flush and submit tasks for
  // execution.
  ASSERT_NO_THROW(delete scope);
}

TEST_F(StreamingChecker, DisallowedOpRelaxed)
{
  auto runtime            = legate::Runtime::get_runtime();
  auto [store_a, store_b] = create_stores();
  const auto* scope       = create_streaming_scope(legate::StreamingMode::RELAXED);

  launch_init_task(store_a, store_b);
  launch_sum_task(store_a, store_b);
  launch_check_task(store_a, store_b);

  runtime->issue_mapping_fence();
  ASSERT_NO_THROW(delete scope);
}

TEST_F(StreamingChecker, DisallowedOpStrict)
{
  auto runtime            = legate::Runtime::get_runtime();
  auto [store_a, store_b] = create_stores();
  const auto* scope       = create_streaming_scope(legate::StreamingMode::STRICT);

  launch_init_task(store_a, store_b);
  launch_sum_task(store_a, store_b);
  launch_check_task(store_a, store_b);

  runtime->issue_mapping_fence();
  ASSERT_THAT(
    [&] {
      delete scope;
      scope = nullptr;
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Failed Streaming Check Disallowed Operation")));
}

TEST_F(StreamingChecker, UnequalLaunchDomainRelaxed)
{
  auto [store_a, store_b] = create_stores();
  const auto* scope       = create_streaming_scope(legate::StreamingMode::RELAXED);

  launch_init_task(store_a, store_b);
  launch_sum_task(store_a, store_b);
  launch_check_task(store_a, store_b);
  launch_manual_init_task(store_a, store_b);
  ASSERT_NO_THROW(delete scope);
}

TEST_F(StreamingChecker, UnequalLaunchDomainStrict)
{
  auto [store_a, store_b] = create_stores();
  const auto* scope       = create_streaming_scope(legate::StreamingMode::STRICT);

  launch_init_task(store_a, store_b);
  launch_sum_task(store_a, store_b);
  launch_check_task(store_a, store_b);
  launch_manual_init_task(store_a, store_b);

  ASSERT_THAT(
    [&] {
      delete scope;
      scope = nullptr;
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Failed Streaming Check Launch Domain Equality")));
}

TEST_F(StreamingChecker, WindowFlushRelaxed)
{
  const auto* scope = create_streaming_scope(legate::StreamingMode::RELAXED);
  ASSERT_NO_THROW(legate::detail::Runtime::get_runtime().flush_scheduling_window());
  ASSERT_NO_THROW(delete scope);
}

TEST_F(StreamingChecker, WindowFlushStrict)
{
  const auto* scope = create_streaming_scope(legate::StreamingMode::STRICT);
  ASSERT_THAT(
    [] {
      legate::detail::Runtime::get_runtime().flush_scheduling_window();
      return 0;
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("flush_scheduling_window called from inside a streaming scope")));
  ASSERT_NO_THROW(delete scope);
}

}  // namespace streaming_checker
