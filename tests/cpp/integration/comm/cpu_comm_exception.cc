/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/comm/coll.h>
#include <legate/comm/detail/local_network.h>
#include <legate/comm/detail/reduction_helpers.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace cpu_comm_exception {

namespace {

constexpr std::size_t SIZE = 100;

class CPUAllreduceExceptionTester : public legate::LegateTask<CPUAllreduceExceptionTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

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

  static void test_and_on_float_internal()
  {
    std::vector<float> send_buffer(5, 1.0F);
    std::vector<float> recv_buffer(5, 0.0F);

    ASSERT_THAT(
      [&]() {
        legate::detail::comm::coll::apply_reduction_typed<float>(
          recv_buffer.data(), send_buffer.data(), 5, legate::ReductionOpKind::AND);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Reduction does not support non-integral types with AND")));
  }

  static void test_or_on_float_internal()
  {
    std::vector<float> send_buffer(5, 1.0F);
    std::vector<float> recv_buffer(5, 0.0F);

    ASSERT_THAT(
      [&]() {
        legate::detail::comm::coll::apply_reduction_typed<float>(
          recv_buffer.data(), send_buffer.data(), 5, legate::ReductionOpKind::OR);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Reduction does not support non-integral types with OR")));
  }

  static void test_xor_on_float_internal()
  {
    std::vector<float> send_buffer(5, 1.0F);
    std::vector<float> recv_buffer(5, 0.0F);

    ASSERT_THAT(
      [&]() {
        legate::detail::comm::coll::apply_reduction_typed<float>(
          recv_buffer.data(), send_buffer.data(), 5, legate::ReductionOpKind::XOR);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Reduction does not support non-integral types with XOR")));
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
    test_and_on_float_internal();
    test_or_on_float_internal();
    test_xor_on_float_internal();
  }
};

class CPUAlltoallvExceptionTester : public legate::LegateTask<CPUAlltoallvExceptionTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void test_null_send_buffer(legate::comm::coll::CollComm comm)
  {
    std::array<int, 2> counts{1, 1};
    std::array<int, 2> displs{0, 1};
    std::array<std::int32_t, 4> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoallv(nullptr,
                      counts.data(),
                      displs.data(),
                      recv_buffer.data(),
                      counts.data(),
                      displs.data(),
                      legate::comm::coll::CollDataType::CollInt,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid sendbuf: nullptr")));
  }

  static void test_null_recv_buffer(legate::comm::coll::CollComm comm)
  {
    std::array<int, 2> counts{1, 1};
    std::array<int, 2> displs{0, 1};
    std::array<std::int32_t, 4> send_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoallv(send_buffer.data(),
                      counts.data(),
                      displs.data(),
                      nullptr,
                      counts.data(),
                      displs.data(),
                      legate::comm::coll::CollDataType::CollInt,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid recvbuf: nullptr")));
  }

  static void test_null_send_counts(legate::comm::coll::CollComm comm)
  {
    std::array<int, 2> displs{0, 1};
    std::array<int, 2> recv_counts{1, 1};
    std::array<std::int32_t, 4> send_buffer{};
    std::array<std::int32_t, 4> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoallv(send_buffer.data(),
                      nullptr,
                      displs.data(),
                      recv_buffer.data(),
                      recv_counts.data(),
                      displs.data(),
                      legate::comm::coll::CollDataType::CollInt,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid sendcounts: nullptr")));
  }

  static void test_null_send_displacements(legate::comm::coll::CollComm comm)
  {
    std::array<int, 2> counts{1, 1};
    std::array<int, 2> recv_counts{1, 1};
    std::array<std::int32_t, 4> send_buffer{};
    std::array<std::int32_t, 4> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoallv(send_buffer.data(),
                      counts.data(),
                      nullptr,
                      recv_buffer.data(),
                      recv_counts.data(),
                      counts.data(),
                      legate::comm::coll::CollDataType::CollInt,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid sdispls: nullptr")));
  }

  static void test_null_recv_counts(legate::comm::coll::CollComm comm)
  {
    std::array<int, 2> send_counts{1, 1};
    std::array<int, 2> displs{0, 1};
    std::array<std::int32_t, 4> send_buffer{};
    std::array<std::int32_t, 4> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoallv(send_buffer.data(),
                      send_counts.data(),
                      displs.data(),
                      recv_buffer.data(),
                      nullptr,
                      displs.data(),
                      legate::comm::coll::CollDataType::CollInt,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid recvcounts: nullptr")));
  }

  static void test_null_recv_displacements(legate::comm::coll::CollComm comm)
  {
    std::array<int, 2> counts{1, 1};
    std::array<std::int32_t, 4> send_buffer{};
    std::array<std::int32_t, 4> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoallv(send_buffer.data(),
                      counts.data(),
                      counts.data(),
                      recv_buffer.data(),
                      counts.data(),
                      nullptr,
                      legate::comm::coll::CollDataType::CollInt,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid rdispls: nullptr")));
  }

  static void test_inplace(legate::comm::coll::CollComm comm)
  {
    std::array<int, 2> counts{1, 1};
    std::array<int, 2> displs{0, 1};
    std::array<std::int32_t, 2> buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoallv(buffer.data(),
                      counts.data(),
                      displs.data(),
                      buffer.data(),
                      counts.data(),
                      displs.data(),
                      legate::comm::coll::CollDataType::CollInt,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Inplace Alltoallv not yet supported")));
  }

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }

    auto comm = context.communicators().at(0).get<legate::comm::coll::CollComm>();

    test_null_send_buffer(comm);
    test_null_recv_buffer(comm);
    test_null_send_counts(comm);
    test_null_send_displacements(comm);
    test_null_recv_counts(comm);
    test_null_recv_displacements(comm);
    test_inplace(comm);
  }
};

class CPUAlltoallExceptionTester : public legate::LegateTask<CPUAlltoallExceptionTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void test_null_send_buffer(legate::comm::coll::CollComm comm)
  {
    std::array<std::int32_t, 2> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoall(
          nullptr, recv_buffer.data(), 1, legate::comm::coll::CollDataType::CollInt, comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid sendbuf: nullptr")));
  }

  static void test_null_recv_buffer(legate::comm::coll::CollComm comm)
  {
    std::array<std::int32_t, 2> send_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoall(
          send_buffer.data(), nullptr, 1, legate::comm::coll::CollDataType::CollInt, comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid recvbuf: nullptr")));
  }

  static void test_invalid_count(legate::comm::coll::CollComm comm)
  {
    std::array<std::int32_t, 2> send_buffer{};
    std::array<std::int32_t, 2> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoall(send_buffer.data(),
                     recv_buffer.data(),
                     0,
                     legate::comm::coll::CollDataType::CollInt,
                     comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("Invalid count: <= 0")));
  }

  static void test_inplace(legate::comm::coll::CollComm comm)
  {
    std::array<std::int32_t, 2> buffer{};

    ASSERT_THAT(
      [&]() {
        collAlltoall(
          buffer.data(), buffer.data(), 1, legate::comm::coll::CollDataType::CollInt, comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Inplace Alltoall not yet supported")));
  }

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }

    auto comm = context.communicators().at(0).get<legate::comm::coll::CollComm>();

    test_null_send_buffer(comm);
    test_null_recv_buffer(comm);
    test_invalid_count(comm);
    test_inplace(comm);
  }
};

class CPUAllgatherExceptionTester : public legate::LegateTask<CPUAllgatherExceptionTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{3}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void test_null_send_buffer(legate::comm::coll::CollComm comm)
  {
    std::array<std::int32_t, 2> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAllgather(
          nullptr, recv_buffer.data(), 1, legate::comm::coll::CollDataType::CollInt, comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid sendbuf: nullptr")));
  }

  static void test_null_recv_buffer(legate::comm::coll::CollComm comm)
  {
    std::array<std::int32_t, 2> send_buffer{};

    ASSERT_THAT(
      [&]() {
        collAllgather(
          send_buffer.data(), nullptr, 1, legate::comm::coll::CollDataType::CollInt, comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("Invalid recvbuf: nullptr")));
  }

  static void test_invalid_count(legate::comm::coll::CollComm comm)
  {
    std::array<std::int32_t, 2> send_buffer{};
    std::array<std::int32_t, 2> recv_buffer{};

    ASSERT_THAT(
      [&]() {
        collAllgather(send_buffer.data(),
                      recv_buffer.data(),
                      0,
                      legate::comm::coll::CollDataType::CollInt,
                      comm);
      },
      testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("Invalid count: <= 0")));
  }

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_TRUE((context.is_single_task() && context.communicators().empty()) ||
                context.communicators().size() == 1);
    if (context.is_single_task()) {
      return;
    }

    auto comm = context.communicators().at(0).get<legate::comm::coll::CollComm>();

    test_null_send_buffer(comm);
    test_null_recv_buffer(comm);
    test_invalid_count(comm);
  }
};

class LocalNetworkAllreduceExceptionTester
  : public legate::LegateTask<LocalNetworkAllreduceExceptionTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{4}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void test_bitwise_ops_on_float(legate::comm::coll::CollComm comm)
  {
    auto* local_network = dynamic_cast<legate::detail::comm::coll::LocalNetwork*>(
      legate::detail::comm::coll::BackendNetwork::get_network().get());

    if (local_network == nullptr) {
      return;
    }

    std::vector<float> send_buffer(5, 1.0F);
    std::vector<float> recv_buffer(5, 0.0F);

    for (auto op : {legate::ReductionOpKind::OR,
                    legate::ReductionOpKind::AND,
                    legate::ReductionOpKind::XOR}) {
      ASSERT_THAT(
        [&]() {
          local_network->all_reduce(send_buffer.data(),
                                    recv_buffer.data(),
                                    recv_buffer.size(),
                                    legate::comm::coll::CollDataType::CollFloat,
                                    op,
                                    comm);
        },
        testing::ThrowsMessage<std::invalid_argument>(
          ::testing::HasSubstr("LocalNetwork::all_reduce does not support float or double "
                               "reduction with bitwise operations")));
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

    test_bitwise_ops_on_float(comm);
  }
};

class ConfigWithExceptions {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_cpu_communicator_exceptions";

  static void registration_callback(legate::Library library)
  {
    CPUAllreduceExceptionTester::register_variants(library);
    CPUAlltoallvExceptionTester::register_variants(library);
    CPUAlltoallExceptionTester::register_variants(library);
    CPUAllgatherExceptionTester::register_variants(library);
    LocalNetworkAllreduceExceptionTester::register_variants(library);
  }
};

class CPUCommunicatorExceptions : public RegisterOnceFixture<ConfigWithExceptions> {};

}  // namespace

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
  runtime->issue_execution_fence(/* block */ true);
}

TEST_F(CPUCommunicatorExceptions, AllgatherExceptionHandling)
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

  auto task = runtime->create_task(context, CPUAllgatherExceptionTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(/* block */ true);
}

TEST_F(CPUCommunicatorExceptions, AlltoallExceptionHandling)
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

  auto task = runtime->create_task(context, CPUAlltoallExceptionTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(/* block */ true);
}

TEST_F(CPUCommunicatorExceptions, AlltoallvExceptionHandling)
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

  auto task = runtime->create_task(context, CPUAlltoallvExceptionTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(/* block */ true);
}

TEST_F(CPUCommunicatorExceptions, LocalNetworkAllreduceExceptionHandling)
{
  constexpr std::int32_t ndim = 3;
  const auto num_procs        = legate::get_machine().count(legate::mapping::TaskTarget::CPU);

  if (num_procs <= 1) {
    GTEST_SKIP() << num_procs;
  }

  auto* local_network = dynamic_cast<legate::detail::comm::coll::LocalNetwork*>(
    legate::detail::comm::coll::BackendNetwork::get_network().get());

  if (local_network == nullptr) {
    GTEST_SKIP() << "LocalNetwork is not available";
  }

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(ConfigWithExceptions::LIBRARY_NAME);
  auto store   = runtime->create_store(
    legate::Shape{
      legate::full<std::uint64_t>(ndim, SIZE)  // NOLINT(readability-suspicious-call-argument)
    },
    legate::int32());

  auto task =
    runtime->create_task(context, LocalNetworkAllreduceExceptionTester::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_output(store, part);
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(/* block */ true);
}

}  // namespace cpu_comm_exception
