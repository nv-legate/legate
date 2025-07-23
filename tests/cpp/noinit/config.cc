/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/config.h>

#include <legate/utilities/detail/env_defaults.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace config_test {

class ConfigTest : public DefaultFixture {};

TEST_F(ConfigTest, DefaultValues)
{
  const legate::detail::Config config;

  ASSERT_TRUE(config.auto_config());
  ASSERT_FALSE(config.show_config());
  ASSERT_FALSE(config.show_progress_requested());
  ASSERT_FALSE(config.use_empty_task());
  ASSERT_FALSE(config.synchronize_stream_view());
  ASSERT_FALSE(config.warmup_nccl());
  ASSERT_FALSE(config.enable_inline_task_launch());
  ASSERT_FALSE(config.profile());
  ASSERT_FALSE(config.provenance());
  ASSERT_EQ(config.num_omp_threads(), 0);
  ASSERT_FALSE(config.show_mapper_usage());
  ASSERT_FALSE(config.need_cuda());
  ASSERT_FALSE(config.need_openmp());
  ASSERT_FALSE(config.need_network());
  ASSERT_EQ(config.max_exception_size(), LEGATE_MAX_EXCEPTION_SIZE_DEFAULT);
  ASSERT_EQ(config.min_cpu_chunk(), LEGATE_MIN_CPU_CHUNK_DEFAULT);
  ASSERT_EQ(config.min_gpu_chunk(), LEGATE_MIN_GPU_CHUNK_DEFAULT);
  ASSERT_EQ(config.min_omp_chunk(), LEGATE_MIN_OMP_CHUNK_DEFAULT);
  ASSERT_EQ(config.window_size(), LEGATE_WINDOW_SIZE_DEFAULT);
  ASSERT_EQ(config.field_reuse_frac(), LEGATE_FIELD_REUSE_FRAC_DEFAULT);
  ASSERT_EQ(config.field_reuse_freq(), LEGATE_FIELD_REUSE_FREQ_DEFAULT);
  ASSERT_EQ(config.consensus(), LEGATE_CONSENSUS_DEFAULT);
  ASSERT_EQ(config.disable_mpi(), LEGATE_DISABLE_MPI_DEFAULT);
  ASSERT_FALSE(config.io_use_vfd_gds());
}

TEST_F(ConfigTest, SetAutoConfig)
{
  legate::detail::Config config;

  config.set_auto_config(false);
  ASSERT_FALSE(config.auto_config());
}

TEST_F(ConfigTest, SetShowConfig)
{
  legate::detail::Config config;

  config.set_show_config(true);
  ASSERT_TRUE(config.show_config());
}

TEST_F(ConfigTest, SetShowProgressRequested)
{
  legate::detail::Config config;

  config.set_show_progress_requested(true);
  ASSERT_TRUE(config.show_progress_requested());
}

TEST_F(ConfigTest, SetUseEmptyTask)
{
  legate::detail::Config config;

  config.set_use_empty_task(true);
  ASSERT_TRUE(config.use_empty_task());
}

TEST_F(ConfigTest, SetSynchronizeStreamView)
{
  legate::detail::Config config;

  config.set_synchronize_stream_view(true);
  ASSERT_TRUE(config.synchronize_stream_view());
}

TEST_F(ConfigTest, SetWarmupNccl)
{
  legate::detail::Config config;

  config.set_warmup_nccl(true);
  ASSERT_TRUE(config.warmup_nccl());
}

TEST_F(ConfigTest, SetEnableInlineTaskLaunch)
{
  legate::detail::Config config;

  config.set_enable_inline_task_launch(true);
  ASSERT_TRUE(config.enable_inline_task_launch());
}

TEST_F(ConfigTest, SetNumOmpThreads)
{
  legate::detail::Config config;
  constexpr std::int64_t test_threads = 42;

  config.set_num_omp_threads(test_threads);
  ASSERT_EQ(config.num_omp_threads(), test_threads);
}

TEST_F(ConfigTest, SetShowMapperUsage)
{
  legate::detail::Config config;

  config.set_show_mapper_usage(true);
  ASSERT_TRUE(config.show_mapper_usage());
}

TEST_F(ConfigTest, SetNeedCuda)
{
  legate::detail::Config config;

  config.set_need_cuda(true);
  ASSERT_TRUE(config.need_cuda());
}

TEST_F(ConfigTest, SetNeedOpenmp)
{
  legate::detail::Config config;

  config.set_need_openmp(true);
  ASSERT_TRUE(config.need_openmp());
}

TEST_F(ConfigTest, SetNeedNetwork)
{
  legate::detail::Config config;

  config.set_need_network(true);
  ASSERT_TRUE(config.need_network());
}

TEST_F(ConfigTest, SetMaxExceptionSize)
{
  legate::detail::Config config;
  constexpr std::uint32_t test_size = 999;

  config.set_max_exception_size(test_size);
  ASSERT_EQ(config.max_exception_size(), test_size);
}

TEST_F(ConfigTest, SetMinCpuChunk)
{
  legate::detail::Config config;
  constexpr std::int64_t test_chunk = 111;

  config.set_min_cpu_chunk(test_chunk);
  ASSERT_EQ(config.min_cpu_chunk(), test_chunk);
}

TEST_F(ConfigTest, SetMinGpuChunk)
{
  legate::detail::Config config;
  constexpr std::int64_t test_chunk = 222;

  config.set_min_gpu_chunk(test_chunk);
  ASSERT_EQ(config.min_gpu_chunk(), test_chunk);
}

TEST_F(ConfigTest, SetMinOmpChunk)
{
  legate::detail::Config config;
  constexpr std::int64_t test_chunk = 333;

  config.set_min_omp_chunk(test_chunk);
  ASSERT_EQ(config.min_omp_chunk(), test_chunk);
}

TEST_F(ConfigTest, SetWindowSize)
{
  legate::detail::Config config;
  constexpr std::uint32_t test_window = 10;

  config.set_window_size(test_window);
  ASSERT_EQ(config.window_size(), test_window);
}

TEST_F(ConfigTest, SetFieldReuseFrac)
{
  legate::detail::Config config;
  constexpr std::uint32_t test_frac = 5;

  config.set_field_reuse_frac(test_frac);
  ASSERT_EQ(config.field_reuse_frac(), test_frac);
}

TEST_F(ConfigTest, SetFieldReuseFreq)
{
  legate::detail::Config config;
  constexpr std::uint32_t test_freq = 3;

  config.set_field_reuse_freq(test_freq);
  ASSERT_EQ(config.field_reuse_freq(), test_freq);
}

TEST_F(ConfigTest, SetConsensus)
{
  legate::detail::Config config;

  config.set_consensus(true);
  ASSERT_TRUE(config.consensus());
}

TEST_F(ConfigTest, SetDisableMpi)
{
  legate::detail::Config config;

  config.set_disable_mpi(true);
  ASSERT_TRUE(config.disable_mpi());
}

TEST_F(ConfigTest, SetIoUseVfdGds)
{
  legate::detail::Config config;

  config.set_io_use_vfd_gds(true);
  ASSERT_TRUE(config.io_use_vfd_gds());
}

}  // namespace config_test
