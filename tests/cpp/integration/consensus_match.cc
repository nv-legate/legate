/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>

#include "core/runtime/detail/runtime.h"
#include "legate.h"

namespace consensus_match {

static const char* library_name = "consensus_match";

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  static_cast<void>(context);
}

struct Thing {
  bool flag;
  int32_t number;
  bool operator==(const Thing& other) const { return flag == other.flag && number == other.number; }
};

TEST(Integration, ConsensusMatch)
{
  auto runtime = legate::Runtime::get_runtime();
  register_tasks();
  auto context = runtime->find_library(library_name);
  static_cast<void>(context);

  Legion::Runtime* legion_runtime = Legion::Runtime::get_runtime();
  Legion::Context legion_context  = legion_runtime->get_context();
  Legion::ShardID sid             = legion_runtime->get_shard_id(legion_context, true);

  std::vector<Thing> input;
  // All shards insert 4 items, but in a different order.
  for (int i = 0; i < 4; ++i) {
    input.emplace_back();
    // Make sure the padding bits have deterministic values. Apparently there is no reliable way to
    // force the compiler to do zero initialization.
    memset(&input.back(), 0, sizeof(Thing));
    switch ((i + sid) % 4) {
      case 0:  // shared between shards
        input.back().flag   = true;
        input.back().number = -1;
        break;
      case 1:  // unique among shards
        input.back().flag   = true;
        input.back().number = sid;
        break;
      case 2:  // shared between shards
        input.back().flag   = false;
        input.back().number = -2;
        break;
      case 3:  // unique among shards
        input.back().flag   = false;
        input.back().number = sid;
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
    Thing ta{true, -1};
    Thing tb{false, -2};
    EXPECT_EQ(result.output().size(), 2);
    EXPECT_TRUE((result.output()[0] == ta && result.output()[1] == tb) ||
                (result.output()[0] == tb && result.output()[1] == ta));
  }
}

}  // namespace consensus_match
