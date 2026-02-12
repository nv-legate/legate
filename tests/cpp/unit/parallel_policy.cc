/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace parallel_policy_test {

using ParallelPolicyTest = DefaultFixture;

namespace {

constexpr std::uint64_t CPU_THRESHOLD = 4;
constexpr std::uint64_t GPU_THRESHOLD = 16;
constexpr std::uint64_t OMP_THRESHOLD = 32;

constexpr std::uint32_t OD_FACTOR = 8;

}  // namespace

TEST_F(ParallelPolicyTest, DefaultConstruct)
{
  const auto pp = legate::ParallelPolicy{};

  ASSERT_EQ(pp.streaming_mode(), legate::StreamingMode::OFF);
  ASSERT_EQ(pp.overdecompose_factor(), 1);

  const auto& cfg = legate::detail::Runtime::get_runtime().config();

  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::CPU), cfg.min_cpu_chunk());
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::GPU), cfg.min_gpu_chunk());
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::OMP), cfg.min_omp_chunk());
}

TEST_F(ParallelPolicyTest, DefaultGlobalScope)
{
  const auto pp = legate::detail::Runtime::get_runtime().scope().parallel_policy();

  ASSERT_EQ(pp.streaming_mode(), legate::StreamingMode::OFF);
  ASSERT_EQ(pp.overdecompose_factor(), 1);

  const auto& cfg = legate::detail::Runtime::get_runtime().config();

  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::CPU), cfg.min_cpu_chunk());
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::GPU), cfg.min_gpu_chunk());
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::OMP), cfg.min_omp_chunk());
}

TEST_F(ParallelPolicyTest, CPUthreshold)
{
  const auto pp = legate::ParallelPolicy{}.with_partitioning_threshold(
    legate::mapping::TaskTarget::CPU, CPU_THRESHOLD);

  const auto& cfg = legate::detail::Runtime::get_runtime().config();

  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::CPU), CPU_THRESHOLD);
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::GPU), cfg.min_gpu_chunk());
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::OMP), cfg.min_omp_chunk());
}

TEST_F(ParallelPolicyTest, GPUthreshold)
{
  const auto pp = legate::ParallelPolicy{}.with_partitioning_threshold(
    legate::mapping::TaskTarget::GPU, GPU_THRESHOLD);

  const auto& cfg = legate::detail::Runtime::get_runtime().config();

  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::CPU), cfg.min_cpu_chunk());
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::GPU), GPU_THRESHOLD);
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::OMP), cfg.min_omp_chunk());
}

TEST_F(ParallelPolicyTest, OMPthreshold)
{
  const auto pp = legate::ParallelPolicy{}.with_partitioning_threshold(
    legate::mapping::TaskTarget::OMP, OMP_THRESHOLD);

  const auto& cfg = legate::detail::Runtime::get_runtime().config();

  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::CPU), cfg.min_cpu_chunk());
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::GPU), cfg.min_gpu_chunk());
  ASSERT_EQ(pp.partitioning_threshold(legate::mapping::TaskTarget::OMP), OMP_THRESHOLD);
}

TEST_F(ParallelPolicyTest, StreamingMode)
{
  const auto pp = legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::RELAXED);

  ASSERT_EQ(pp.streaming_mode(), legate::StreamingMode::RELAXED);
  ASSERT_EQ(pp.overdecompose_factor(), 1);
}

TEST_F(ParallelPolicyTest, ODfactor)
{
  const auto pp = legate::ParallelPolicy{}
                    .with_streaming(legate::StreamingMode::STRICT)
                    .with_overdecompose_factor(OD_FACTOR);
  ;

  ASSERT_EQ(pp.streaming_mode(), legate::StreamingMode::STRICT);
  ASSERT_EQ(pp.overdecompose_factor(), OD_FACTOR);
}

}  // namespace parallel_policy_test
