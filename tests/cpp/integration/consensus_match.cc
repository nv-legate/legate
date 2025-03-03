/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <cstring>
#include <utilities/utilities.h>

namespace consensus_match {

using Integration = DefaultFixture;

namespace {

constexpr std::string_view LIBRARY_NAME = "consensus_match";

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(LIBRARY_NAME);
  static_cast<void>(context);
}

struct Thing {
  bool flag;
  std::int32_t number;
  bool operator==(const Thing& other) const { return flag == other.flag && number == other.number; }
};

}  // namespace

TEST_F(Integration, ConsensusMatch)
{
  auto runtime = legate::Runtime::get_runtime();
  register_tasks();
  auto context = runtime->find_library(LIBRARY_NAME);
  static_cast<void>(context);

  Legion::Runtime* legion_runtime = Legion::Runtime::get_runtime();
  Legion::Context legion_context  = Legion::Runtime::get_context();
  const Legion::ShardID sid       = legion_runtime->get_shard_id(legion_context, true);

  std::vector<Thing> input;
  // All shards insert 4 items, but in a different order.
  for (int i = 0; i < 4; ++i) {
    input.emplace_back();
    // Make sure the padding bits have deterministic values. Apparently there is no reliable way to
    // force the compiler to do zero initialization.
    std::memset(&input.back(), 0, sizeof(Thing));
    // We silence the lint below because 0-3 covers any % 4
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch ((i + sid) % 4) {
      case 0:  // shared between shards
        input.back().flag   = true;
        input.back().number = -1;
        break;
      case 1:  // unique among shards
        input.back().flag   = true;
        input.back().number = static_cast<std::int32_t>(sid);
        break;
      case 2:  // shared between shards
        input.back().flag   = false;
        input.back().number = -2;
        break;
      case 3:  // unique among shards
        input.back().flag   = false;
        input.back().number = static_cast<std::int32_t>(sid);
        break;
    }
  }

  auto result = runtime->impl()->issue_consensus_match(std::move(input));
  result.wait();

  if (legion_runtime->get_num_shards(legion_context, true) < 2) {
    EXPECT_EQ(result.output().size(), 4);
    EXPECT_EQ(result.output()[0], result.input()[0]);
    EXPECT_EQ(result.output()[1], result.input()[1]);
    EXPECT_EQ(result.output()[2], result.input()[2]);
    EXPECT_EQ(result.output()[3], result.input()[3]);
  } else {
    const Thing ta{true, -1};
    const Thing tb{false, -2};

    EXPECT_EQ(result.output().size(), 2);
    EXPECT_TRUE((result.output()[0] == ta && result.output()[1] == tb) ||
                (result.output()[0] == tb && result.output()[1] == ta));
  }
}

}  // namespace consensus_match
