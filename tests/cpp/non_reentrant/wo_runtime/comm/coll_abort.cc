/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/coll.h>
#include <legate/comm/detail/local_network.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace coll_abort_test {

using CollAbortDeathTest = ::testing::Test;
using legate::detail::comm::coll::BackendNetwork;

TEST_F(CollAbortDeathTest, CollAbortWithNetwork)
{
  BackendNetwork::create_network(std::make_unique<legate::detail::comm::coll::LocalNetwork>());

  ASSERT_EXIT({ legate::detail::comm::coll::abort(); }, ::testing::KilledBySignal{SIGABRT}, "");
}

TEST_F(CollAbortDeathTest, CollAbortWithoutNetwork)
{
  ASSERT_EXIT({ legate::detail::comm::coll::abort(); }, ::testing::KilledBySignal{SIGABRT}, "");
}

}  // namespace coll_abort_test
