/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/backend_network.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utilities/utilities.h>

namespace backend_network_test {

using legate::detail::comm::coll::BackendNetwork;
using BackendNetworkTest = ::testing::Test;

// Test that attempting to get the network before initialization throws an exception
TEST(BackendNetworkTest, GetNetworkBeforeInitialization)
{
  ASSERT_FALSE(BackendNetwork::has_network());
  ASSERT_THAT([&] { static_cast<void>(BackendNetwork::get_network()); },
              ::testing::ThrowsMessage<std::logic_error>(::testing::HasSubstr(
                "Trying to retrieve backend network before it has been initialized. Call "
                "BackendNetwork::create_network() first")));
}

}  // namespace backend_network_test
