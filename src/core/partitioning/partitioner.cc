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
#include "core/data/scalar.h"
#include "core/partitioning/partition.h"
#include "core/runtime/launcher.h"
#include "core/runtime/operation.h"
#include "core/runtime/runtime.h"

namespace legate {

Strategy::Strategy() : launch_domain_(nullptr) {}

Strategy::Strategy(const Legion::Domain& launch_domain)
  : launch_domain_(std::make_unique<Legion::Domain>(launch_domain))
{
}

bool Strategy::parallel() const { return nullptr != launch_domain_; }

Legion::Domain Strategy::launch_domain() const { return *launch_domain_; }

void Strategy::set_launch_domain(const Legion::Domain& launch_domain)
{
  launch_domain_ = std::make_unique<Legion::Domain>(launch_domain);
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

Partitioner::Partitioner(Runtime* runtime, std::vector<const Operation*>&& operations)
  : runtime_(runtime), operations_(std::forward<std::vector<const Operation*>>(operations))
{
}

std::unique_ptr<Strategy> Partitioner::partition_stores()
{
  auto strategy = std::make_unique<Strategy>();

  for (auto op : operations_) {
    auto all_stores = op->all_stores();
    for (auto store : all_stores) strategy->insert(store, create_no_partition(runtime_));
  }

  return std::move(strategy);
}

}  // namespace legate
