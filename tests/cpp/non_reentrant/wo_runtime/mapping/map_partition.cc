/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/detail/partition/image.h>

#include <gtest/gtest.h>

#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_map_partition {

namespace {

class OpenMPMapPartition : public DefaultFixture {
 protected:
  void SetUp() override
  {
    ASSERT_NO_THROW(legate::start());
    DefaultFixture::SetUp();
  }

  void TearDown() override
  {
    DefaultFixture::TearDown();
    ASSERT_EQ(legate::finish(), 0);
  }

 private:
  // Legate starts its implicit top-level task on a LOC_PROC, so keep one CPU for startup
  // and pass an OMP-only machine to the partition operation below.
  legate::test::Environment::TemporaryEnvVar legate_config_{
    /*name=*/"LEGATE_CONFIG",
    /*value=*/
    "--auto-config 0 --cpus 1 --gpus 0 --omps 1 --ompthreads 1 --numamem 256 "
    "--sysmem 4000",
    /*overwrite=*/true};
};

}  // namespace

TEST_F(OpenMPMapPartition, ConstructsImagePartitionWithOpenMPConfigured)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto machine  = runtime->get_machine();

  ASSERT_GT(machine.count(legate::mapping::TaskTarget::OMP), 0);
  ASSERT_EQ(machine.count(legate::mapping::TaskTarget::CPU), 1);
  ASSERT_EQ(machine.count(legate::mapping::TaskTarget::GPU), 0);

  const auto local_machine = legate::mapping::detail::LocalMachine{};
  ASSERT_TRUE(local_machine.has_omps());
  ASSERT_EQ(local_machine.cpus().size(), 1);
  ASSERT_FALSE(local_machine.has_gpus());
  ASSERT_EQ(local_machine.omps().front().kind(), Legion::Processor::Kind::OMP_PROC);

  const auto omp_machine = machine.only(legate::mapping::TaskTarget::OMP);

  ASSERT_EQ(omp_machine.count(legate::mapping::TaskTarget::CPU), 0);
  ASSERT_EQ(omp_machine.count(legate::mapping::TaskTarget::GPU), 0);
  ASSERT_EQ(omp_machine.count(legate::mapping::TaskTarget::OMP),
            machine.count(legate::mapping::TaskTarget::OMP));

  auto func  = runtime->create_store(legate::Shape{2, 2, 2}, legate::point_type(3));
  auto range = runtime->create_store(legate::Shape{4, 4, 4}, legate::int64());

  runtime->issue_fill(func, legate::Scalar{legate::Point<3>{0, 0, 0}});
  runtime->issue_execution_fence(/*block=*/true);

  auto func_partition  = func.partition_by_tiling({1, 1, 1});
  auto image_partition = legate::detail::create_image(func.impl(),
                                                      func_partition.impl()->partition(),
                                                      *omp_machine.impl(),
                                                      legate::ImageComputationHint::NO_HINT);
  auto logical_partition =
    image_partition->construct(range.impl()->get_region_field()->region(), /*complete=*/false);

  ASSERT_NE(logical_partition, Legion::LogicalPartition::NO_PART);
  runtime->issue_execution_fence(/*block=*/true);
}

}  // namespace test_map_partition
