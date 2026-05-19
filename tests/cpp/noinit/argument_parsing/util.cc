/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/util.h>

#include <gtest/gtest.h>

#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_argument_parsing_util {

namespace {

class NumRanksUnit : public DefaultFixture {
 protected:
  using Environment = legate::test::Environment;

  Environment::TemporaryEnvVar ompi_comm_world_size_ =
    Environment::temporary_cleared_env_var("OMPI_COMM_WORLD_SIZE");
  Environment::TemporaryEnvVar mv2_comm_world_size_ =
    Environment::temporary_cleared_env_var("MV2_COMM_WORLD_SIZE");
  Environment::TemporaryEnvVar slurm_ntasks_ =
    Environment::temporary_cleared_env_var("SLURM_NTASKS");
};

TEST_F(NumRanksUnit, DefaultsToSingleRank)
{
  ASSERT_EQ(legate::detail::num_ranks(), 1);
  ASSERT_FALSE(legate::detail::multi_node_job());
}

TEST_F(NumRanksUnit, UsesOMPIWhenPresent)
{
  const auto ompi =
    Environment::temporary_env_var("OMPI_COMM_WORLD_SIZE", /*value=*/"8", /*overwrite=*/true);
  const auto mv2 =
    Environment::temporary_env_var("MV2_COMM_WORLD_SIZE", /*value=*/"4", /*overwrite=*/true);
  const auto slurm =
    Environment::temporary_env_var("SLURM_NTASKS", /*value=*/"16", /*overwrite=*/true);

  ASSERT_EQ(legate::detail::num_ranks(), 8);
  ASSERT_TRUE(legate::detail::multi_node_job());
}

TEST_F(NumRanksUnit, FallsBackToMV2)
{
  const auto ompi =
    Environment::temporary_env_var("OMPI_COMM_WORLD_SIZE", /*value=*/"1", /*overwrite=*/true);
  const auto mv2 =
    Environment::temporary_env_var("MV2_COMM_WORLD_SIZE", /*value=*/"4", /*overwrite=*/true);

  ASSERT_EQ(legate::detail::num_ranks(), 4);
}

TEST_F(NumRanksUnit, FallsBackToSLURM)
{
  const auto ompi =
    Environment::temporary_env_var("OMPI_COMM_WORLD_SIZE", /*value=*/"1", /*overwrite=*/true);
  const auto mv2 =
    Environment::temporary_env_var("MV2_COMM_WORLD_SIZE", /*value=*/"1", /*overwrite=*/true);
  const auto slurm =
    Environment::temporary_env_var("SLURM_NTASKS", /*value=*/"3", /*overwrite=*/true);

  ASSERT_EQ(legate::detail::num_ranks(), 3);
}

}  // namespace

}  // namespace test_argument_parsing_util
