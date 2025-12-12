/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/oob_allgather.h>

#include <legate/comm/detail/mpi_interface.h>
#include <legate/comm/detail/mpi_oob_allgather.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <thread>
#include <utilities/utilities.h>
#include <vector>

namespace oob_allgather_test {

using legate::detail::comm::coll::MPIOOBAllgather;
using legate::detail::comm::mpi::detail::MPIInterface;

// Test fixture for OOB allgather tests
class OOBAllgatherTest : public DefaultFixture {};

// Test construction - MPIOOBAllgather can be created and used
TEST_F(OOBAllgatherTest, DefaultConstruction)
{
  int init = 0;
  MPIInterface::mpi_initialized(&init);
  if (!init) {
    GTEST_SKIP() << "MPI not initialized";
  }

  // Test that we can create an MPIOOBAllgather instance
  const std::vector<int> mapping_table = {0, 0};
  auto mpi_oob_allgather               = std::make_unique<MPIOOBAllgather>(0, 2, mapping_table);
  void* req                            = nullptr;  // NOLINT(misc-const-correctness)

  // Test member functions with invalid arguments - should fail gracefully
  ASSERT_NE(mpi_oob_allgather->allgather(nullptr, nullptr, 0, nullptr, &req), UCC_OK);
  ASSERT_EQ(mpi_oob_allgather->test(nullptr), UCC_OK);  // MPI impl returns OK for test
  ASSERT_EQ(mpi_oob_allgather->free(nullptr), UCC_OK);  // MPI impl returns OK for free
}

// Test type erasure with mock implementation
TEST_F(OOBAllgatherTest, MPIAllgather)
{
  int init = 0;
  MPIInterface::mpi_initialized(&init);
  if (!init) {
    GTEST_SKIP() << "MPI not initialized";
  }

  const std::vector<int> mapping_table = {0, 0};
  std::unique_ptr<MPIOOBAllgather> mpi_oob_allgather =
    std::make_unique<MPIOOBAllgather>(0, 2, mapping_table);
  std::unique_ptr<MPIOOBAllgather> mpi_oob_allgather2 =
    std::make_unique<MPIOOBAllgather>(1, 2, mapping_table);

  std::vector<int> recv_data(2);
  void* req = nullptr;

  std::thread t{[&]() {
    const int send_data = 42;

    const ucc_status_t status = mpi_oob_allgather->allgather(
      &send_data, recv_data.data(), sizeof(int), /*allgather_info=*/nullptr, &req);

    EXPECT_EQ(status, UCC_OK);

    // Test that test and free also return UCC_OK
    EXPECT_EQ(mpi_oob_allgather->test(req), UCC_OK);
    EXPECT_EQ(mpi_oob_allgather->free(req), UCC_OK);
  }};

  std::thread t2{[&]() {
    const int send_data = 54;

    const ucc_status_t status = mpi_oob_allgather2->allgather(
      &send_data, recv_data.data(), sizeof(int), /*allgather_info=*/nullptr, &req);

    EXPECT_EQ(status, UCC_OK);

    // Test that test and free also return UCC_OK
    EXPECT_EQ(mpi_oob_allgather2->test(req), UCC_OK);
    EXPECT_EQ(mpi_oob_allgather2->free(req), UCC_OK);
  }};

  t.join();
  t2.join();

  EXPECT_EQ(recv_data[0], 42);
  EXPECT_EQ(recv_data[1], 54);
  mpi_oob_allgather.reset();
  mpi_oob_allgather2.reset();
}

}  // namespace oob_allgather_test
