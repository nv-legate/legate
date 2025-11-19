/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/operation/detail/discard.h>
#include <legate/operation/detail/fill.h>
#include <legate/operation/detail/task.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/detail/streaming/analysis.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_process_streaming_run {

namespace {

class DummyTask : public legate::LegateTask<DummyTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1));

  static void cpu_variant(legate::TaskContext) {}
};

[[nodiscard]] legate::LogicalStore make_store(const legate::Shape& shape)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto ret            = runtime->create_store(shape, legate::int32());

  runtime->issue_fill(ret, legate::Scalar{std::int32_t{0}});
  return ret;
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_process_streaming_run";

  static void registration_callback(legate::Library library)
  {
    DummyTask::register_variants(library);
  }
};

// Something high enough where we can be sure the runtime never reaches it.
constexpr std::uint64_t DUMMY_OP_ID     = 999999999;
constexpr std::uint64_t DUMMY_UNIQUE_ID = 999999999;

}  // namespace

class ProcessStreamingRunUnit : public RegisterOnceFixture<Config> {};

TEST_F(ProcessStreamingRunUnit, Empty)
{
  std::deque<legate::InternalSharedPtr<legate::detail::Operation>> run;

  ASSERT_NO_THROW(legate::detail::process_streaming_run(&run));
  ASSERT_THAT(run, ::testing::IsEmpty());
}

TEST_F(ProcessStreamingRunUnit, Basic)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, DummyTask::TASK_CONFIG.task_id());
  const auto store_a  = make_store(legate::Shape{3});
  const auto store_b  = make_store(legate::Shape{4});

  task.add_input(store_a);
  task.add_output(store_b);

  const auto discard = [&] {
    auto&& rf = store_a.impl()->get_region_field();

    return legate::make_internal_shared<legate::detail::Discard>(
      DUMMY_OP_ID, rf->region(), rf->field_id());
  }();

  // The kind of internal operation here is not important, only that it isn't a user task.
  const auto fill =
    legate::make_internal_shared<legate::detail::Fill>(store_a.impl(),
                                                       store_b.impl(),
                                                       DUMMY_UNIQUE_ID,
                                                       /* priority */ 0,
                                                       runtime->impl()->get_machine());

  auto ops = std::deque<legate::InternalSharedPtr<legate::detail::Operation>>{
    task.impl_(),  // A regular user task
    fill,          // This should be ignored
    discard        // This should cause some discarded stores
  };

  ASSERT_EQ(task.impl_()->inputs().at(0).privilege, LEGION_READ_ONLY);
  ASSERT_EQ(task.impl_()->outputs().at(0).privilege, LEGION_WRITE_ONLY);

  legate::detail::process_streaming_run(&ops);

  ASSERT_EQ(task.impl_()->inputs().at(0).privilege, LEGION_READ_ONLY | LEGION_DISCARD_OUTPUT_MASK);
  // Only the first one should be modified
  ASSERT_EQ(task.impl_()->outputs().at(0).privilege, LEGION_WRITE_ONLY);
  ASSERT_THAT(ops, ::testing::ElementsAre(task.impl_(), fill));
}

TEST_F(ProcessStreamingRunUnit, BasicDiscardFirst)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, DummyTask::TASK_CONFIG.task_id());
  const auto store_a  = make_store(legate::Shape{3});
  const auto store_b  = make_store(legate::Shape{4});

  task.add_input(store_a);
  task.add_output(store_b);

  const auto discard = [&] {
    auto&& rf = store_a.impl()->get_region_field();

    return legate::make_internal_shared<legate::detail::Discard>(
      DUMMY_OP_ID, rf->region(), rf->field_id());
  }();

  // The kind of internal operation here is not important, only that it isn't a user task.
  const auto fill =
    legate::make_internal_shared<legate::detail::Fill>(store_a.impl(),
                                                       store_b.impl(),
                                                       DUMMY_UNIQUE_ID,
                                                       /* priority */ 0,
                                                       runtime->impl()->get_machine());

  auto ops = std::deque<legate::InternalSharedPtr<legate::detail::Operation>>{
    discard,  // Because this is first, it should have no effect
    task.impl_(),
    fill};

  ASSERT_EQ(task.impl_()->inputs().at(0).privilege, LEGION_READ_ONLY);
  ASSERT_EQ(task.impl_()->outputs().at(0).privilege, LEGION_WRITE_ONLY);

  legate::detail::process_streaming_run(&ops);

  // Discard came before the task, so nothing should change
  ASSERT_EQ(task.impl_()->inputs().at(0).privilege, LEGION_READ_ONLY);
  ASSERT_EQ(task.impl_()->outputs().at(0).privilege, LEGION_WRITE_ONLY);
  ASSERT_THAT(ops, ::testing::ElementsAre(discard, task.impl_(), fill));
}

TEST_F(ProcessStreamingRunUnit, NoDiscards)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(lib, DummyTask::TASK_CONFIG.task_id());
  const auto store_a  = make_store(legate::Shape{3});
  const auto store_b  = make_store(legate::Shape{4});

  task.add_input(store_a);
  task.add_output(store_b);

  auto ops = std::deque<legate::InternalSharedPtr<legate::detail::Operation>>{task.impl_()};

  ASSERT_EQ(task.impl_()->inputs().at(0).privilege, LEGION_READ_ONLY);
  ASSERT_EQ(task.impl_()->outputs().at(0).privilege, LEGION_WRITE_ONLY);

  legate::detail::process_streaming_run(&ops);

  // No discards, so nothing should change
  ASSERT_EQ(task.impl_()->inputs().at(0).privilege, LEGION_READ_ONLY);
  ASSERT_EQ(task.impl_()->outputs().at(0).privilege, LEGION_WRITE_ONLY);
  ASSERT_THAT(ops, ::testing::ElementsAre(task.impl_()));
}

TEST_F(ProcessStreamingRunUnit, OnlyDiscards)
{
  const auto store_a = make_store(legate::Shape{3});
  const auto discard = [&] {
    auto&& rf = store_a.impl()->get_region_field();

    return legate::make_internal_shared<legate::detail::Discard>(
      DUMMY_OP_ID, rf->region(), rf->field_id());
  }();

  auto ops = std::deque<legate::InternalSharedPtr<legate::detail::Operation>>{discard};

  // Nothing should happen here
  ASSERT_NO_THROW(legate::detail::process_streaming_run(&ops));
  ASSERT_THAT(ops, ::testing::ElementsAre(discard));
}

}  // namespace test_process_streaming_run
