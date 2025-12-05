/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/local_machine_selector.h>

#include <legate.h>

#include <legate/mapping/detail/machine.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace global_machine_test {

using LocalMachineSelectorTest = DefaultFixture;

TEST_F(LocalMachineSelectorTest, ProcCounts)
{
  // Compare LocalMachine returned by selector in comparison to one allocated as is.
  auto selector             = legate::mapping::detail::LocalMachineSelector{};
  const auto& local_machine = selector.get_local();
  const legate::mapping::detail::LocalMachine reference_local_machine{};

  if (local_machine.has_cpus()) {
    ASSERT_EQ(local_machine.total_cpu_count(), reference_local_machine.total_cpu_count());
  }
  if (local_machine.has_gpus()) {
    ASSERT_EQ(local_machine.total_gpu_count(), reference_local_machine.total_gpu_count());
  }
  if (local_machine.has_omps()) {
    ASSERT_EQ(local_machine.total_omp_count(), reference_local_machine.total_omp_count());
  }
}

TEST_F(LocalMachineSelectorTest, LocalToProc)
{
  // Ensure caching mechanism works and LocalMachine is not re-constructed unnecessarily.
  auto selector             = legate::mapping::detail::LocalMachineSelector{};
  const auto& local_machine = selector.get_local();

  if (local_machine.has_cpus()) {
    const Legion::Processor cpu = local_machine.cpus().front();

    ASSERT_EQ(&local_machine, &selector.get_local_to(cpu));
  } else {
    const legate::mapping::detail::LocalMachine reference_local_machine{};

    ASSERT_TRUE(local_machine.cpus().empty() == reference_local_machine.cpus().empty());
  }
}

TEST_F(LocalMachineSelectorTest, LocalToMemory)
{
  // Ensure caching mechanism works and LocalMachine is not re-constructed unnecessarily.
  auto selector               = legate::mapping::detail::LocalMachineSelector{};
  const auto& local_machine   = selector.get_local();
  const Legion::Memory memory = local_machine.system_memory();

  ASSERT_EQ(&local_machine, &selector.get_local_to(memory));
}

}  // namespace global_machine_test
