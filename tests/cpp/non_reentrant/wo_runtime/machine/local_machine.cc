/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/mapping/detail/machine.h>

#include <gtest/gtest.h>

#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_local_machine {

namespace {

class CPUOnlyMachine : public DefaultFixture {
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
  legate::test::Environment::TemporaryEnvVar legate_config_{
    "LEGATE_CONFIG",
    /*value=*/"--auto-config 0 --cpus 2 --omps 0 --numamem 0 --sysmem 4000",
    /*overwrite=*/true};
};

class OpenMPWithoutNUMAMemory : public DefaultFixture {
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
  legate::test::Environment::TemporaryEnvVar legate_config_{
    "LEGATE_CONFIG",
    /*value=*/"--auto-config 0 --cpus 2 --omps 1 --ompthreads 1 --numamem 0 --sysmem 4000",
    /*overwrite=*/true};
};

class OpenMPWithNUMAMemory : public DefaultFixture {
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
  legate::test::Environment::TemporaryEnvVar legate_config_{
    "LEGATE_CONFIG",
    /*value=*/"--auto-config 0 --cpus 2 --omps 1 --ompthreads 1 --numamem 256 --sysmem 4000",
    /*overwrite=*/true};
};

class GPUWithZeroCopyMemory : public DefaultFixture {
 protected:
  void SetUp() override
  {
    if (!LEGATE_DEFINED(LEGATE_USE_CUDA)) {
      GTEST_SKIP() << "GPU support is required to get zero-copy memory";
    }

    ASSERT_NO_THROW(legate::start());
    runtime_started_ = true;
    DefaultFixture::SetUp();
  }

  void TearDown() override
  {
    if (!runtime_started_) {
      return;
    }

    DefaultFixture::TearDown();
    ASSERT_EQ(legate::finish(), 0);
  }

 private:
  bool runtime_started_{false};
  legate::test::Environment::TemporaryEnvVar legate_config_{
    "LEGATE_CONFIG",
    /*value=*/"--auto-config 0 --cpus 2 --gpus 1 --fbmem 1000 --zcmem 256 --sysmem 4000",
    /*overwrite=*/true};
};

}  // namespace

TEST_F(CPUOnlyMachine, HasNoSocketMemory)
{
  const auto local_machine = legate::mapping::detail::LocalMachine{};

  ASSERT_FALSE(local_machine.has_omps());
  ASSERT_TRUE(local_machine.socket_memories().empty());
  ASSERT_FALSE(local_machine.has_socket_memory());
}

TEST_F(OpenMPWithoutNUMAMemory, UsesSystemMemoryForOpenMPProcessors)
{
  const auto local_machine = legate::mapping::detail::LocalMachine{};
  const auto system_memory = local_machine.system_memory();

  ASSERT_TRUE(local_machine.has_omps());
  ASSERT_FALSE(local_machine.has_socket_memory());
  ASSERT_EQ(local_machine.socket_memories().size(), local_machine.omps().size());

  for (auto&& [omp, memory] : local_machine.socket_memories()) {
    ASSERT_EQ(memory, system_memory);
    ASSERT_EQ(local_machine.get_memory(omp, legate::mapping::StoreTarget::SOCKETMEM),
              system_memory);
    ASSERT_EQ(local_machine.get_memory(omp, legate::Memory::Kind::SOCKET_MEM), system_memory);
  }
}

TEST_F(OpenMPWithNUMAMemory, ReportsTotalSocketMemorySize)
{
  constexpr std::uint32_t FIELD_REUSE_FRAC = 2;
  const auto local_machine                 = legate::mapping::detail::LocalMachine{};

  ASSERT_TRUE(local_machine.has_omps());
  ASSERT_TRUE(local_machine.has_socket_memory());
  ASSERT_FALSE(local_machine.socket_memories().empty());

  const auto expected_per_node_size = local_machine.socket_memories().size() *
                                      local_machine.socket_memories().begin()->second.capacity();
  const auto expected_total_socket_memory_size = expected_per_node_size * local_machine.total_nodes;

  ASSERT_EQ(local_machine.total_socket_memory_size(), expected_total_socket_memory_size);
  ASSERT_EQ(local_machine.calculate_field_reuse_size(FIELD_REUSE_FRAC),
            expected_total_socket_memory_size / FIELD_REUSE_FRAC);
  ASSERT_EQ(
    local_machine.get_memory(local_machine.omps().front(), legate::Memory::Kind::SOCKET_MEM),
    local_machine.socket_memories().at(local_machine.omps().front()));
}

TEST_F(GPUWithZeroCopyMemory, GetsZeroCopyMemoryForGPU)
{
  const auto local_machine = legate::mapping::detail::LocalMachine{};

  if (!local_machine.has_gpus()) {
    GTEST_SKIP() << "GPU processors are required to get zero-copy memory";
  }

  ASSERT_EQ(
    local_machine.get_memory(local_machine.gpus().front(), legate::mapping::StoreTarget::ZCMEM),
    local_machine.zerocopy_memory());
  ASSERT_EQ(
    local_machine.get_memory(local_machine.gpus().front(), legate::Memory::Kind::GPU_FB_MEM),
    local_machine.frame_buffers().at(local_machine.gpus().front()));
  ASSERT_EQ(
    local_machine.get_memory(local_machine.gpus().front(), legate::Memory::Kind::Z_COPY_MEM),
    local_machine.zerocopy_memory());
  ASSERT_EQ(
    local_machine.g2c_multi_hop_bandwidth(
      local_machine.frame_buffers().at(local_machine.gpus().front()), legate::Memory::NO_MEMORY),
    0);
}

}  // namespace test_local_machine
