/* Copyright 2021 NVIDIA Corporation
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

#include "core/partitioning/partitioner.h"
#include "core/data/logical_store.h"
#include "core/data/scalar.h"
#include "core/partitioning/partition.h"
#include "core/runtime/launcher.h"
#include "core/runtime/operation.h"
#include "core/runtime/runtime.h"

namespace legate {

Strategy::Strategy() {}

bool Strategy::parallel(const Operation* op) const
{
  auto finder = launch_domains_.find(op);
  return finder != launch_domains_.end();
}

Legion::Domain Strategy::launch_domain(const Operation* op) const
{
  auto finder = launch_domains_.find(op);
  assert(finder != launch_domains_.end());
  return finder->second;
}

void Strategy::set_launch_domain(const Operation* op, const Legion::Domain& launch_domain)
{
  launch_domains_[op] = launch_domain;
}

void Strategy::insert(const LogicalStore* store, std::shared_ptr<Partition> partition)
{
  assert(assignments_.find(store) == assignments_.end());
  assignments_[store] = std::move(partition);
}

std::shared_ptr<Partition> Strategy::find(const LogicalStore* store) const
{
  auto finder = assignments_.find(store);
  assert(finder != assignments_.end());
  return finder->second;
}

std::unique_ptr<Projection> Strategy::get_projection(LogicalStore* store) const
{
  auto partition = find(store);
  return partition->get_projection(store);
}

Partitioner::Partitioner(Runtime* runtime, std::vector<Operation*>&& operations)
  : runtime_(runtime), operations_(std::forward<std::vector<Operation*>>(operations))
{
}

std::unique_ptr<Strategy> Partitioner::partition_stores()
{
  auto strategy = std::make_unique<Strategy>();

  for (auto op : operations_) {
    bool determined_launch_domain = false;
    auto all_stores               = op->all_stores();
    for (auto store : all_stores) {
      auto key_partition = store->find_or_create_key_partition();
      if (!determined_launch_domain) {
        determined_launch_domain = true;
        if (key_partition->has_launch_domain())
          strategy->set_launch_domain(op, key_partition->launch_domain());
      }
      strategy->insert(store, std::move(key_partition));
    }
  }

  return std::move(strategy);
}

}  // namespace legate
