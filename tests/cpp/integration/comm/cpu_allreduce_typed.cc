/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/comm/coll.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <integration/comm/common_comm.h>
#include <type_traits>
#include <utilities/utilities.h>
#include <vector>

namespace cpu_allreduce_typed {

namespace {

template <typename T>
class CPUAllreduceTypedTester : public legate::LegateTask<CPUAllreduceTypedTester<T>> {
 public:
  // Each type gets a unique task ID: 0 for int8, 1 for char, etc.
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{[]() {
      if constexpr (std::is_same_v<T, std::int8_t>) {
        return 0;
      } else if constexpr (std::is_same_v<T, char>) {
        return 1;
      } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return 2;
      } else if constexpr (std::is_same_v<T, int>) {
        return 3;
      } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return 4;
      } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return 5;
      } else if constexpr (std::is_same_v<T, std::uint64_t>) {
        return 6;
      } else if constexpr (std::is_same_v<T, float>) {
        return 7;
      } else if constexpr (std::is_same_v<T, double>) {
        return 8;
      }
    }()}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);
  static constexpr auto COLL_TYPE           = common_comm::TypeToCollDataType<T>::VALUE;

  static void test_sum(legate::comm::coll::CollComm comm,
                       std::int64_t num_tasks,
                       std::int64_t task_index)
  {
    constexpr std::size_t count = 5;

    // Each task contributes (task_index + 1) to the reduction
    std::vector<T> send_buffer(count, static_cast<T>(task_index + 1));
    std::vector<T> recv_buffer(count, static_cast<T>(0));

    collAllreduce(send_buffer.data(),
                  recv_buffer.data(),
                  count,
                  COLL_TYPE,  // Use the type-specific CollDataType
                  legate::ReductionOpKind::ADD,
                  comm);

    // Expected sum: 1 + 2 + ... + num_tasks = num_tasks * (num_tasks + 1) / 2
    const T expected_sum = static_cast<T>(num_tasks * (num_tasks + 1)) / static_cast<T>(2);

    ASSERT_THAT(recv_buffer, ::testing::Each(expected_sum));
  }

  static void test_max(legate::comm::coll::CollComm comm,
                       std::int64_t num_tasks,
                       std::int64_t task_index)
  {
    constexpr std::size_t count = 4;
    std::vector<T> send_buffer(count, static_cast<T>(task_index));
    std::vector<T> recv_buffer(count, static_cast<T>(0));

    collAllreduce(send_buffer.data(),
                  recv_buffer.data(),
                  count,
                  COLL_TYPE,  // Use the type-specific CollDataType
                  legate::ReductionOpKind::MAX,
                  comm);

    // Expected max: num_tasks - 1 (highest task index)
    const T expected_max = static_cast<T>(num_tasks - 1);

    ASSERT_THAT(recv_buffer, ::testing::Each(expected_max));
  }

  static void test_min(legate::comm::coll::CollComm comm, std::int64_t task_index)
  {
    constexpr std::size_t count = 4;
    std::vector<T> send_buffer(count, static_cast<T>(task_index));
    std::vector<T> recv_buffer(count, static_cast<T>(0));

    collAllreduce(send_buffer.data(),
                  recv_buffer.data(),
                  count,
                  COLL_TYPE,  // Use the type-specific CollDataType
                  legate::ReductionOpKind::MIN,
                  comm);

    // Expected min: 0 (lowest task index)
    constexpr T expected_min = static_cast<T>(0);

    ASSERT_THAT(recv_buffer, ::testing::Each(expected_min));
  }

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }

    auto comm             = context.communicator(0).get<legate::comm::coll::CollComm>();
    const auto num_tasks  = context.get_launch_domain().get_volume();
    const auto task_index = comm->global_rank;

    test_sum(comm, static_cast<std::int64_t>(num_tasks), task_index);
    test_max(comm, static_cast<std::int64_t>(num_tasks), task_index);
    test_min(comm, task_index);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_cpu_allreduce_typed";

  static void registration_callback(legate::Library library)
  {
    // Register all typed variants
    CPUAllreduceTypedTester<std::int8_t>::register_variants(library);
    CPUAllreduceTypedTester<char>::register_variants(library);
    CPUAllreduceTypedTester<std::uint8_t>::register_variants(library);
    CPUAllreduceTypedTester<int>::register_variants(library);
    CPUAllreduceTypedTester<std::uint32_t>::register_variants(library);
    CPUAllreduceTypedTester<std::int64_t>::register_variants(library);
    CPUAllreduceTypedTester<std::uint64_t>::register_variants(library);
    CPUAllreduceTypedTester<float>::register_variants(library);
    CPUAllreduceTypedTester<double>::register_variants(library);
  }
};

using AllSupportedTypes = ::testing::Types<std::int8_t,
                                           char,
                                           std::uint8_t,
                                           int,
                                           std::uint32_t,
                                           std::int64_t,
                                           std::uint64_t,
                                           float,
                                           double>;

template <typename T>
void trigger_reduction_typed_task()
{
  const auto num_procs = legate::get_machine().count(legate::mapping::TaskTarget::CPU);

  if (num_procs <= 1) {
    GTEST_SKIP() << "Skipping test: requires more than 1 CPU processor, found " << num_procs;
  }

  constexpr std::int32_t ndim = 3;
  constexpr std::size_t SIZE  = 100;

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());

  auto task = runtime->create_task(context, CPUAllreduceTypedTester<T>::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(/* block */ true);
}

}  // namespace

template <typename T>
class CPUAllreduceTypedTest : public RegisterOnceFixture<Config> {};

TYPED_TEST_SUITE(CPUAllreduceTypedTest, AllSupportedTypes, );

TYPED_TEST(CPUAllreduceTypedTest, Reduction) { trigger_reduction_typed_task<TypeParam>(); }

}  // namespace cpu_allreduce_typed
