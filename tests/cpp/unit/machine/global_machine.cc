/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/global_machine.h>

#include <legate.h>

#include <legate/mapping/detail/machine.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace global_machine_test {

using GlobalMachineTest = DefaultFixture;

class GlobalMachineProcTest : public GlobalMachineTest,
                              public ::testing::WithParamInterface<legate::mapping::TaskTarget> {};

INSTANTIATE_TEST_SUITE_P(GlobalMachineTest,
                         GlobalMachineProcTest,
                         ::testing::Values(legate::mapping::TaskTarget::CPU,
                                           legate::mapping::TaskTarget::GPU,
                                           legate::mapping::TaskTarget::OMP));

TEST_P(GlobalMachineProcTest, Procs)
{
  const auto global_machine = legate::mapping::detail::GlobalMachine{};
  const auto local_machine  = legate::mapping::detail::LocalMachine{};
  const auto target         = GetParam();
  const auto global_count   = global_machine.procs(target).size();

  ASSERT_TRUE(global_count % global_machine.total_nodes() == 0);
  ASSERT_TRUE(global_count / global_machine.total_nodes() == local_machine.procs(target).size());
}

TEST_F(GlobalMachineTest, SliceCPU)
{
  const auto global_machine = legate::mapping::detail::GlobalMachine{};
  const auto local_machine  = legate::mapping::detail::LocalMachine{};
  const auto machine =
    legate::Runtime::get_runtime()->get_machine().impl()->only(legate::mapping::TaskTarget::CPU);
  const auto sliced = global_machine.slice(machine);

  if (local_machine.has_cpus()) {
    const auto& cpus = local_machine.cpus();

    ASSERT_FALSE(sliced.empty());
    ASSERT_THAT(cpus, ::testing::Contains(sliced.first()));
  } else {
    ASSERT_TRUE(sliced.empty());
  }

  ASSERT_EQ(sliced.total_proc_count(), local_machine.total_cpu_count());
}

TEST_F(GlobalMachineTest, SliceEmpty)
{
  const auto global_machine = legate::mapping::detail::GlobalMachine{};
  const auto local_machine  = legate::mapping::detail::LocalMachine{};
  const auto empty_targets  = legate::Span<const legate::mapping::TaskTarget>{};
  const auto machine = legate::Runtime::get_runtime()->get_machine().impl()->only(empty_targets);
  const auto sliced  = global_machine.slice(machine);

  ASSERT_TRUE(sliced.empty());
}

TEST_F(GlobalMachineTest, SliceCPUAndGPU)
{
  const auto global_machine = legate::mapping::detail::GlobalMachine{};
  const auto local_machine  = legate::mapping::detail::LocalMachine{};
  const auto machine        = legate::Runtime::get_runtime()->get_machine().impl()->only(
    {legate::mapping::TaskTarget::CPU, legate::mapping::TaskTarget::GPU});
  const auto slice = global_machine.slice(machine);

  if (local_machine.has_gpus()) {
    const auto& gpus = local_machine.gpus();

    ASSERT_FALSE(slice.empty());
    ASSERT_THAT(gpus, ::testing::Contains(slice.first()));
  } else if (local_machine.has_cpus()) {
    const auto& cpus = local_machine.cpus();

    ASSERT_FALSE(slice.empty());
    ASSERT_THAT(cpus, ::testing::Contains(slice.first()));
  } else {
    ASSERT_TRUE(slice.empty());
  }
}

TEST_F(GlobalMachineTest, SliceFallbackCPU)
{
  const auto global_machine = legate::mapping::detail::GlobalMachine{};
  const auto local_machine  = legate::mapping::detail::LocalMachine{};
  const auto machine =
    legate::Runtime::get_runtime()->get_machine().impl()->only(legate::mapping::TaskTarget::CPU);
  const auto sliced_global = global_machine.slice_with_fallback(machine);

  if (local_machine.has_cpus()) {
    const auto& cpus = local_machine.cpus();

    ASSERT_FALSE(sliced_global.empty());
    ASSERT_THAT(cpus, ::testing::Contains(sliced_global.first()));
  } else {
    ASSERT_TRUE(sliced_global.empty());
  }

  ASSERT_EQ(sliced_global.total_proc_count(), local_machine.total_cpu_count());
}

TEST_F(GlobalMachineTest, SliceFallbackEmpty)
{
  const auto global_machine = legate::mapping::detail::GlobalMachine{};
  const auto local_machine  = legate::mapping::detail::LocalMachine{};
  const auto empty_targets  = legate::Span<const legate::mapping::TaskTarget>{};
  const auto machine = legate::Runtime::get_runtime()->get_machine().impl()->only(empty_targets);
  const auto target  = machine.preferred_target();
  const auto sliced  = global_machine.slice_with_fallback(machine);

  if (target == legate::mapping::TaskTarget::CPU && local_machine.has_cpus()) {
    ASSERT_FALSE(sliced.empty());
  } else if (target == legate::mapping::TaskTarget::GPU && local_machine.has_gpus()) {
    ASSERT_FALSE(sliced.empty());
  } else if (target == legate::mapping::TaskTarget::OMP && local_machine.has_omps()) {
    ASSERT_FALSE(sliced.empty());
  } else {
    ASSERT_TRUE(sliced.empty());
  }
}

TEST_F(GlobalMachineTest, SliceFallbackCPUAndGPU)
{
  const auto global_machine = legate::mapping::detail::GlobalMachine{};
  const auto local_machine  = legate::mapping::detail::LocalMachine{};
  const auto machine        = legate::Runtime::get_runtime()->get_machine().impl()->only(
    {legate::mapping::TaskTarget::CPU, legate::mapping::TaskTarget::GPU});
  const auto slice = global_machine.slice_with_fallback(machine);

  if (local_machine.has_gpus()) {
    const auto& gpus = local_machine.gpus();

    ASSERT_FALSE(slice.empty());
    ASSERT_THAT(gpus, ::testing::Contains(slice.first()));
  } else if (local_machine.has_cpus()) {
    const auto& cpus = local_machine.cpus();

    ASSERT_FALSE(slice.empty());
    ASSERT_THAT(cpus, ::testing::Contains(slice.first()));
  } else {
    // Empty fallback here because we already know that there are no processors in this machine.
    ASSERT_TRUE(slice.empty());
  }
}

}  // namespace global_machine_test
