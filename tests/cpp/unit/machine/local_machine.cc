/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/mapping/detail/machine.h>
#include <legate/mapping/machine.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace local_machine_test {

using LocalMachineTest        = DefaultFixture;
using LocalProcessorRangeTest = DefaultFixture;

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

TEST_F(LocalMachineTest, SliceCPU)
{
  auto local_machine = legate::mapping::detail::LocalMachine{};
  auto sliced        = local_machine.slice(legate::mapping::TaskTarget::CPU,
                                    *legate::Runtime::get_runtime()->get_machine().impl());
  auto sliced_global = local_machine.slice(legate::mapping::TaskTarget::CPU,
                                           *legate::Runtime::get_runtime()->get_machine().impl(),
                                           true /* fallback_to_global */);

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
  auto sliced        = local_machine.slice(legate::mapping::TaskTarget::GPU,
                                    *legate::Runtime::get_runtime()->get_machine().impl());
  auto sliced_global = local_machine.slice(legate::mapping::TaskTarget::GPU,
                                           *legate::Runtime::get_runtime()->get_machine().impl(),
                                           true /* fallback_to_global */);

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
  auto sliced        = local_machine.slice(legate::mapping::TaskTarget::OMP,
                                    *legate::Runtime::get_runtime()->get_machine().impl());
  auto sliced_global = local_machine.slice(legate::mapping::TaskTarget::OMP,
                                           *legate::Runtime::get_runtime()->get_machine().impl(),
                                           true /* fallback_to_global */);

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

TEST_F(LocalProcessorRangeTest, Create)
{
  auto local_processor_range = legate::mapping::detail::LocalProcessorRange{};
  ASSERT_TRUE(local_processor_range.empty());
  ASSERT_EQ(local_processor_range.to_string(),
            "{offset: 0, total processor count: 0, processors: }");
}

}  // namespace local_machine_test
