/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/comm/coll.h>

#include <gtest/gtest.h>

#include <cmath>
#include <utilities/utilities.h>

namespace cpu_communicator {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::size_t SIZE = 100;

struct CPUCommunicatorTester : public legate::LegateTask<CPUCommunicatorTester> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }
    auto comm = context.communicators().at(0).get<legate::comm::coll::CollComm>();

    constexpr std::int64_t value = 12345;
    const auto num_tasks         = context.get_launch_domain().get_volume();
    std::vector<std::int64_t> recv_buffer(num_tasks, 0);
    collAllgather(&value, recv_buffer.data(), 1, legate::comm::coll::CollDataType::CollInt64, comm);
    for (auto v : recv_buffer) {
      EXPECT_EQ(v, value);
    }
  }
};

class CPUAllreduceTester : public legate::LegateTask<CPUAllreduceTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void test_sum(legate::comm::coll::CollComm comm,
                       std::int64_t num_tasks,
                       std::int64_t task_index)
  {
    constexpr std::size_t count = 5;
    std::vector<std::int64_t> send_buffer(count, task_index + 1);
    std::vector<std::int64_t> recv_buffer(count, 0);

    collAllreduce(send_buffer.data(),
                  recv_buffer.data(),
                  count,
                  legate::comm::coll::CollDataType::CollInt64,
                  legate::ReductionOpKind::ADD,
                  comm);

    // Expected sum: 1 + 2 + ... + num_tasks = num_tasks * (num_tasks + 1) / 2
    const std::int64_t expected_sum = (num_tasks * (num_tasks + 1)) / 2;
    for (auto v : recv_buffer) {
      ASSERT_EQ(v, expected_sum);
    }
  }

  static void test_prod(legate::comm::coll::CollComm comm, std::int64_t num_tasks)
  {
    constexpr std::size_t count = 3;
    std::vector<double> send_buffer(count, 2.0);
    std::vector<double> recv_buffer(count, 0.0);

    collAllreduce(send_buffer.data(),
                  recv_buffer.data(),
                  count,
                  legate::comm::coll::CollDataType::CollDouble,
                  legate::ReductionOpKind::MUL,
                  comm);

    // Expected product: 2^num_tasks
    const double expected_prod = 1 << num_tasks;

    for (auto v : recv_buffer) {
      ASSERT_DOUBLE_EQ(v, expected_prod);
    }
  }

  static void test_max(legate::comm::coll::CollComm comm,
                       std::int64_t num_tasks,
                       std::int64_t task_index)
  {
    constexpr std::size_t count = 4;
    std::vector<std::int32_t> send_buffer(count, static_cast<std::int32_t>(task_index));
    std::vector<std::int32_t> recv_buffer(count, 0);

    collAllreduce(send_buffer.data(),
                  recv_buffer.data(),
                  count,
                  legate::comm::coll::CollDataType::CollInt,
                  legate::ReductionOpKind::MAX,
                  comm);

    // Expected max: num_tasks - 1 (highest task index)
    const auto expected_max = static_cast<std::int32_t>(num_tasks - 1);

    for (auto v : recv_buffer) {
      ASSERT_EQ(v, expected_max);
    }
  }

  static void test_min(legate::comm::coll::CollComm comm, std::int64_t task_index)
  {
    constexpr std::size_t count = 4;
    std::vector<std::int32_t> send_buffer(count, static_cast<std::int32_t>(task_index));
    std::vector<std::int32_t> recv_buffer(count, 0);

    collAllreduce(send_buffer.data(),
                  recv_buffer.data(),
                  count,
                  legate::comm::coll::CollDataType::CollInt,
                  legate::ReductionOpKind::MIN,
                  comm);

    // Expected min: 0 (lowest task index)
    constexpr std::int32_t expected_min = 0;

    for (auto v : recv_buffer) {
      ASSERT_EQ(v, expected_min);
    }
  }

  static void test_bor(legate::comm::coll::CollComm comm,
                       std::int64_t num_tasks,
                       std::int64_t task_index)
  {
    constexpr std::size_t count = 1;
    // Each rank sets a bit
    const std::int32_t rank_bit = 1 << task_index;
    std::vector<std::int32_t> send_buffer(count, rank_bit);
    std::vector<std::int32_t> recv_buffer(count, 0);

    collAllreduce(send_buffer.data(),
                  recv_buffer.data(),
                  count,
                  legate::comm::coll::CollDataType::CollInt,
                  legate::ReductionOpKind::OR,
                  comm);

    // Expected: all lower num_tasks bits should be set
    const std::int32_t expected_bor = (1 << num_tasks) - 1;

    for (auto v : recv_buffer) {
      ASSERT_EQ(v, expected_bor);
    }
  }

  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }

    auto comm             = context.communicators().at(0).get<legate::comm::coll::CollComm>();
    const auto num_tasks  = context.get_launch_domain().get_volume();
    const auto task_index = comm->global_rank;

    test_sum(comm, static_cast<std::int64_t>(num_tasks), task_index);
    test_prod(comm, static_cast<std::int64_t>(num_tasks));
    test_max(comm, static_cast<std::int64_t>(num_tasks), task_index);
    test_min(comm, task_index);
    test_bor(comm, static_cast<std::int64_t>(num_tasks), task_index);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_cpu_communicator";

  static void registration_callback(legate::Library library)
  {
    CPUCommunicatorTester::register_variants(library);
    CPUAllreduceTester::register_variants(library);
  }
};

class CPUCommunicator : public RegisterOnceFixture<Config> {};

class CPUCommunicatorParameterized : public CPUCommunicator,
                                     public testing::WithParamInterface<std::int32_t> {};

void test_cpu_communicator_auto(std::int32_t ndim)
{
  const auto num_procs = legate::get_machine().count(legate::mapping::TaskTarget::CPU);

  if (num_procs <= 1) {
    GTEST_SKIP() << num_procs;
  }

  auto runtime = legate::Runtime::get_runtime();

  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());

  auto task = runtime->create_task(context, CPUCommunicatorTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();
  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

void test_cpu_communicator_manual(std::int32_t ndim)
{
  const auto num_procs = legate::get_machine().count(legate::mapping::TaskTarget::CPU);

  if (num_procs <= 1) {
    GTEST_SKIP() << num_procs;
  }

  auto runtime = legate::Runtime::get_runtime();

  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());
  auto launch_shape = legate::full<std::uint64_t>(ndim, 1);
  auto tile_shape   = legate::full<std::uint64_t>(ndim, 1);

  const auto num_tasks = std::min<std::size_t>(num_procs, SIZE);
  launch_shape[0]      = num_tasks;
  tile_shape[0]        = (SIZE + num_tasks - 1) / num_tasks;

  auto part = store.partition_by_tiling(tile_shape.data());

  auto task =
    runtime->create_task(context, CPUCommunicatorTester::TASK_CONFIG.task_id(), launch_shape);
  task.add_output(part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

void test_cpu_allreduce_auto(std::int32_t ndim)
{
  const auto num_procs = legate::get_machine().count(legate::mapping::TaskTarget::CPU);

  if (num_procs <= 1) {
    GTEST_SKIP() << num_procs;
  }

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());

  auto task = runtime->create_task(context, CPUAllreduceTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

void test_cpu_allreduce_manual(std::int32_t ndim)
{
  const auto num_procs = legate::get_machine().count(legate::mapping::TaskTarget::CPU);

  if (num_procs <= 1) {
    GTEST_SKIP() << num_procs;
  }

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());
  auto launch_shape    = legate::full<std::uint64_t>(ndim, 1);
  auto tile_shape      = legate::full<std::uint64_t>(ndim, 1);
  const auto num_tasks = std::min<std::size_t>(num_procs, SIZE);

  launch_shape[0] = num_tasks;
  tile_shape[0]   = (SIZE + num_tasks - 1) / num_tasks;

  auto part = store.partition_by_tiling(tile_shape.data());
  auto task =
    runtime->create_task(context, CPUAllreduceTester::TASK_CONFIG.task_id(), launch_shape);

  task.add_output(part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

}  // namespace

// Test case with single unbound store
// TODO(jfaibussowit)
// Currently causes unexplained hangs in CI. To be fixed by
// https://github.com/nv-legate/legate.internal/pull/700
TEST_F(CPUCommunicator, DISABLED_ALL)
{
  for (auto ndim : {1, 3}) {
    test_cpu_communicator_auto(ndim);
    test_cpu_communicator_manual(ndim);
  }
}

TEST_P(CPUCommunicatorParameterized, AllreduceAutoTask)
{
  const auto ndim = GetParam();

  test_cpu_allreduce_auto(ndim);
}

TEST_P(CPUCommunicatorParameterized, AllreduceManualTask)
{
  const auto ndim = GetParam();

  test_cpu_allreduce_manual(ndim);
}

INSTANTIATE_TEST_SUITE_P(CPUCommunicatorTests, CPUCommunicatorParameterized, testing::Values(1, 3));

class CPUAllreduceExceptionTester : public legate::LegateTask<CPUAllreduceExceptionTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void test_invalid_count(legate::comm::coll::CollComm comm)
  {
    std::vector<std::int32_t> send_buffer(5, 1);
    std::vector<std::int32_t> recv_buffer(5, 0);

    ASSERT_THAT(
      [&]() {
        collAllreduce(send_buffer.data(),
                      recv_buffer.data(),
                      0,
                      legate::comm::coll::CollDataType::CollInt,
                      legate::ReductionOpKind::ADD,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("Invalid count: <= 0")));
  }

  static void test_null_send_buffer(legate::comm::coll::CollComm comm)
  {
    std::vector<std::int32_t> recv_buffer(5, 0);

    ASSERT_THAT(
      [&]() {
        collAllreduce(nullptr,
                      recv_buffer.data(),
                      recv_buffer.size(),
                      legate::comm::coll::CollDataType::CollInt,
                      legate::ReductionOpKind::ADD,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid sendbuf: nullptr")));
  }

  static void test_null_recv_buffer(legate::comm::coll::CollComm comm)
  {
    std::vector<std::int32_t> send_buffer(5, 1);

    ASSERT_THAT(
      [&]() {
        collAllreduce(send_buffer.data(),
                      nullptr,  // Invalid: null pointer
                      send_buffer.size(),
                      legate::comm::coll::CollDataType::CollInt,
                      legate::ReductionOpKind::ADD,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid recvbuf: nullptr")));
  }

  static void test_bitwise_ops_on_float(legate::comm::coll::CollComm comm)
  {
    std::vector<float> send_buffer(5, 1.0F);
    std::vector<float> recv_buffer(5, 0.0F);

    for (auto op : {legate::ReductionOpKind::OR,
                    legate::ReductionOpKind::AND,
                    legate::ReductionOpKind::XOR}) {
      ASSERT_THAT(
        [&]() {
          collAllreduce(send_buffer.data(),
                        recv_buffer.data(),
                        recv_buffer.size(),
                        legate::comm::coll::CollDataType::CollFloat,
                        op,
                        comm);
        },
        testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
          "all_reduce does not support float or double reduction with bitwise operations")));
    }
  }

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }

    auto comm = context.communicators().at(0).get<legate::comm::coll::CollComm>();

    test_invalid_count(comm);
    test_null_send_buffer(comm);
    test_null_recv_buffer(comm);
    test_bitwise_ops_on_float(comm);
  }
};

class ConfigWithExceptions {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_cpu_communicator_exceptions";

  static void registration_callback(legate::Library library)
  {
    CPUAllreduceExceptionTester::register_variants(library);
  }
};

class CPUCommunicatorExceptions : public RegisterOnceFixture<ConfigWithExceptions> {};

TEST_F(CPUCommunicatorExceptions, AllreduceExceptionHandling)
{
  constexpr std::int32_t ndim = 3;
  const auto num_procs        = legate::get_machine().count(legate::mapping::TaskTarget::CPU);

  if (num_procs <= 1) {
    GTEST_SKIP() << num_procs;
  }

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(ConfigWithExceptions::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());

  auto task = runtime->create_task(context, CPUAllreduceExceptionTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(true);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace cpu_communicator
