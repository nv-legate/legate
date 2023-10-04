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

#include "core/data/detail/logical_store.h"
#include "core/runtime/detail/runtime.h"
#include "legate.h"
#include "utilities/utilities.h"

namespace field_reuse {

using Integration = DefaultFixture;

static const char* library_name = "field_reuse";

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  static_cast<void>(context);
}

void check_field_is_new(Legion::FieldID fid)
{
  static std::set<Legion::FieldID> unique_fields;
  size_t prev_size = unique_fields.size();
  unique_fields.insert(fid);
  EXPECT_EQ(unique_fields.size(), prev_size + 1);
}

TEST_F(Integration, FieldReuse)
{
  // TODO: Also test the reuse of a field originally returned by an unbounded-output task.

  auto runtime = legate::Runtime::get_runtime();
  register_tasks();
  auto context = runtime->find_library(library_name);
  static_cast<void>(context);

  Legion::Runtime* legion_runtime = Legion::Runtime::get_runtime();
  Legion::Context legion_context  = legion_runtime->get_context();
  Legion::ShardID sid             = legion_runtime->get_shard_id(legion_context, true);
  std::vector<legate::LogicalStore> shard_local_stores;

  if (legion_runtime->get_num_shards(legion_context, true) < 2) return;

  uint32_t field_reuse_freq = runtime->impl()->field_reuse_freq();
  EXPECT_GE(field_reuse_freq, 7);  // otherwise the consensus match would be triggered too early
  uint32_t num_allocations = 0;
  auto make_store          = [&]() {
    ++num_allocations;
    return runtime->create_store({5, 5}, legate::int64());
  };

  // A Store freed in-order will be reused immediately.
  Legion::FieldID fid1;
  {
    auto store = make_store();
    fid1       = store.impl()->get_region_field()->field_id();
    check_field_is_new(fid1);
  }
  auto store1 = make_store();
  EXPECT_EQ(fid1, store1.impl()->get_region_field()->field_id());

  // This store is marked for out-of-order destruction, but all shards actually free it in-order.
  Legion::FieldID fid2;
  {
    auto store = make_store();
    store.impl()->allow_out_of_order_destruction();
    fid2 = store.impl()->get_region_field()->field_id();
    check_field_is_new(fid2);
  }

  // This store is only freed on even shards.
  Legion::FieldID fid3;
  {
    auto store = make_store();
    store.impl()->allow_out_of_order_destruction();
    fid3 = store.impl()->get_region_field()->field_id();
    check_field_is_new(fid3);
    if (sid % 2 == 0) shard_local_stores.push_back(store);
  }

  // This store is only freed on odd shards.
  Legion::FieldID fid4;
  {
    auto store = make_store();
    store.impl()->allow_out_of_order_destruction();
    fid4 = store.impl()->get_region_field()->field_id();
    check_field_is_new(fid4);
    if (sid % 2 == 1) shard_local_stores.push_back(store);
  }

  // This store is kept alive on all shards.
  Legion::FieldID fid5;
  {
    auto store = make_store();
    store.impl()->allow_out_of_order_destruction();
    fid5 = store.impl()->get_region_field()->field_id();
    check_field_is_new(fid5);
    shard_local_stores.push_back(store);
  }

  // None of the previous 4 fields should be reusable yet, so the next allocation will need to
  // create a new field. Free and reuse this field enough times to trigger a consensus match.
  Legion::FieldID fid6 = 0;
  while (num_allocations % field_reuse_freq != 0) {
    auto store = make_store();
    if (fid6 == 0) {
      fid6 = store.impl()->get_region_field()->field_id();
      check_field_is_new(fid6);
    } else
      EXPECT_EQ(fid6, store.impl()->get_region_field()->field_id());
  }

  // At this point the consensus match has been triggered, but fid6 is still available, so the next
  // allocation will just reuse that.
  auto store6 = make_store();
  EXPECT_EQ(fid6, store6.impl()->get_region_field()->field_id());

  // No more in-order-freed fields remain, so we will block on the consensus match, and reuse the
  // only field that was universally freed.
  auto store2 = make_store();
  EXPECT_EQ(fid2, store2.impl()->get_region_field()->field_id());

  // Free any locally cached fields, so they will be included in the next consensus match.
  shard_local_stores.clear();

  // The next allocation will need to create a new field. Free and reuse this field enough times to
  // trigger a consensus match.
  Legion::FieldID fid7 = 0;
  while (num_allocations % field_reuse_freq != 0) {
    auto store = make_store();
    if (fid7 == 0) {
      fid7 = store.impl()->get_region_field()->field_id();
      check_field_is_new(fid7);
    } else
      EXPECT_EQ(fid7, store.impl()->get_region_field()->field_id());
  }

  // At this point the consensus match has been triggered, but fid7 is still available, so the next
  // allocation will just reuse that.
  auto store7 = make_store();
  EXPECT_EQ(fid7, store7.impl()->get_region_field()->field_id());

  // No more in-order-freed fields remain, so we will block on the consensus match. The next three
  // allocations should reuse fid3, fid4 and fid5, but in an undefined order.
  std::vector<legate::LogicalStore> stores345 = {make_store(), make_store(), make_store()};
  std::set<Legion::FieldID> fields345;
  for (const auto& store : stores345)
    fields345.insert(store.impl()->get_region_field()->field_id());
  EXPECT_TRUE(fields345.count(fid3) && fields345.count(fid4) && fields345.count(fid5));

  // The next allocation will need to create a new field.
  auto store8          = make_store();
  Legion::FieldID fid8 = store8.impl()->get_region_field()->field_id();
  check_field_is_new(fid8);
}

}  // namespace field_reuse
