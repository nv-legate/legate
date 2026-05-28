/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/mapping/detail/machine.h>
#include <legate/mapping/machine.h>

#include <gtest/gtest.h>

#include <sstream>
#include <utilities/utilities.h>

namespace local_machine_test {

using LocalMachineTest  = DefaultFixture;
using ProcessorSpanTest = DefaultFixture;

namespace {

void assert_same_local_machine(const legate::mapping::detail::LocalMachine& result,
                               const legate::mapping::detail::LocalMachine& expected)
{
  ASSERT_EQ(result.node_id, expected.node_id);
  ASSERT_EQ(result.total_nodes, expected.total_nodes);
  ASSERT_EQ(result.cpus(), expected.cpus());
  ASSERT_EQ(result.gpus(), expected.gpus());
  ASSERT_EQ(result.omps(), expected.omps());
  ASSERT_EQ(result.system_memory(), expected.system_memory());
  ASSERT_EQ(result.zerocopy_memory(), expected.zerocopy_memory());
  ASSERT_EQ(result.frame_buffers(), expected.frame_buffers());
  ASSERT_EQ(result.socket_memories(), expected.socket_memories());
}

}  // namespace

TEST_F(LocalMachineTest, CPU)
{
  auto local_machine = legate::mapping::detail::LocalMachine{};

  if (local_machine.has_cpus()) {
    ASSERT_TRUE(local_machine.has_cpus());
    ASSERT_GT(local_machine.cpus().size(), 0);
    ASSERT_EQ(local_machine.total_cpu_count(),
              local_machine.total_nodes * local_machine.cpus().size());
  } else {
    ASSERT_FALSE(local_machine.has_cpus());
    ASSERT_TRUE(local_machine.cpus().empty());
    ASSERT_EQ(local_machine.total_cpu_count(), 0);
  }
}

TEST_F(LocalMachineTest, GPU)
{
  auto local_machine = legate::mapping::detail::LocalMachine{};

  if (local_machine.has_gpus()) {
    ASSERT_TRUE(local_machine.has_gpus());
    ASSERT_GT(local_machine.gpus().size(), 0);
    ASSERT_EQ(local_machine.total_gpu_count(),
              local_machine.total_nodes * local_machine.gpus().size());
  } else {
    ASSERT_FALSE(local_machine.has_gpus());
    ASSERT_TRUE(local_machine.gpus().empty());
    ASSERT_EQ(local_machine.total_gpu_count(), 0);
  }
}

TEST_F(LocalMachineTest, OMP)
{
  auto local_machine = legate::mapping::detail::LocalMachine{};

  if (local_machine.has_omps()) {
    ASSERT_TRUE(local_machine.has_omps());
    ASSERT_GT(local_machine.omps().size(), 0);
    ASSERT_EQ(local_machine.total_omp_count(),
              local_machine.total_nodes * local_machine.omps().size());
  } else {
    ASSERT_FALSE(local_machine.has_omps());
    ASSERT_TRUE(local_machine.omps().empty());
    ASSERT_EQ(local_machine.total_omp_count(), 0);
  }
}

TEST_F(LocalMachineTest, ProcessorAccessors)
{
  using LocalMachine = legate::mapping::detail::LocalMachine;
  using Accessor     = const std::vector<legate::Processor>& (LocalMachine::*)() const;

  const auto local_machine = LocalMachine{};
  const Accessor cpus      = &LocalMachine::cpus;
  const Accessor gpus      = &LocalMachine::gpus;
  const Accessor omps      = &LocalMachine::omps;

  ASSERT_EQ((local_machine.*cpus)(), local_machine.cpus());
  ASSERT_EQ((local_machine.*gpus)(), local_machine.gpus());
  ASSERT_EQ((local_machine.*omps)(), local_machine.omps());
}

TEST_F(LocalMachineTest, CreateFromProcessor)
{
  const auto local_machine = legate::mapping::detail::LocalMachine{};

  if (!local_machine.has_cpus()) {
    GTEST_SKIP() << "CPU processors are required to construct LocalMachine from a processor";
  }

  assert_same_local_machine(legate::mapping::detail::LocalMachine{local_machine.cpus().front()},
                            local_machine);
}

TEST_F(LocalMachineTest, CreateFromMemory)
{
  const auto local_machine = legate::mapping::detail::LocalMachine{};

  assert_same_local_machine(legate::mapping::detail::LocalMachine{local_machine.system_memory()},
                            local_machine);
}

TEST_F(LocalMachineTest, SliceCPU)
{
  auto local_machine = legate::mapping::detail::LocalMachine{};
  auto machine =
    legate::Runtime::get_runtime()->get_machine().impl()->only(legate::mapping::TaskTarget::CPU);
  auto sliced        = local_machine.slice(machine);
  auto sliced_global = local_machine.slice_with_fallback(machine);

  if (local_machine.has_cpus()) {
    ASSERT_FALSE(sliced.empty());
    ASSERT_FALSE(sliced_global.empty());
  } else {
    ASSERT_TRUE(sliced.empty());
    ASSERT_TRUE(sliced_global.empty());
  }

  ASSERT_EQ(sliced.total_proc_count(), local_machine.total_cpu_count());
  ASSERT_EQ(sliced_global.total_proc_count(), local_machine.total_cpu_count());
}

TEST_F(LocalMachineTest, SliceGPU)
{
  auto local_machine = legate::mapping::detail::LocalMachine{};
  auto machine =
    legate::Runtime::get_runtime()->get_machine().impl()->only(legate::mapping::TaskTarget::GPU);
  auto sliced        = local_machine.slice(machine);
  auto sliced_global = local_machine.slice_with_fallback(machine);

  if (local_machine.has_gpus()) {
    ASSERT_FALSE(sliced.empty());
    ASSERT_FALSE(sliced_global.empty());
  } else {
    ASSERT_TRUE(sliced.empty());
    ASSERT_TRUE(sliced_global.empty());
  }

  ASSERT_EQ(sliced.total_proc_count(), local_machine.total_gpu_count());
  ASSERT_EQ(sliced_global.total_proc_count(), local_machine.total_gpu_count());
}

TEST_F(LocalMachineTest, SliceOMP)
{
  auto local_machine = legate::mapping::detail::LocalMachine{};
  auto machine =
    legate::Runtime::get_runtime()->get_machine().impl()->only(legate::mapping::TaskTarget::OMP);
  auto sliced        = local_machine.slice(machine);
  auto sliced_global = local_machine.slice_with_fallback(machine);

  if (local_machine.has_omps()) {
    ASSERT_FALSE(sliced.empty());
    ASSERT_FALSE(sliced_global.empty());
  } else {
    ASSERT_TRUE(sliced.empty());
    ASSERT_TRUE(sliced_global.empty());
  }

  ASSERT_EQ(sliced.total_proc_count(), local_machine.total_omp_count());
  ASSERT_EQ(sliced_global.total_proc_count(), local_machine.total_omp_count());
}

TEST_F(LocalMachineTest, FindProcessor)
{
  auto local_machine = legate::mapping::detail::LocalMachine{};

  if (!local_machine.has_cpus()) {
    ASSERT_THAT(
      [&] {
        static_cast<void>(local_machine.find_first_processor_with_affinity_to(
          legate::mapping::StoreTarget::SYSMEM));
      },
      ::testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("No CPU processors exist to satisfy store target")));
  }

  if (!local_machine.has_gpus()) {
    ASSERT_THAT(
      [&] {
        static_cast<void>(
          local_machine.find_first_processor_with_affinity_to(legate::mapping::StoreTarget::FBMEM));
      },
      ::testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("No GPU processors exist to satisfy store target")));
  }

  if (!local_machine.has_omps()) {
    ASSERT_THAT(
      [&] {
        static_cast<void>(local_machine.find_first_processor_with_affinity_to(
          legate::mapping::StoreTarget::SOCKETMEM));
      },
      ::testing::ThrowsMessage<std::invalid_argument>(
        ::testing::HasSubstr("No OpenMP processors exist to satisfy")));
  }
}

TEST_F(ProcessorSpanTest, Create)
{
  auto processor_span = legate::mapping::detail::ProcessorSpan{};
  ASSERT_TRUE(processor_span.empty());
  ASSERT_EQ(processor_span.to_string(), "{offset: 0, total processor count: 0, processors: }");
}

}  // namespace local_machine_test
