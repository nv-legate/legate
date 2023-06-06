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

#include "core/data/detail/logical_region_field.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

LogicalRegionField::LogicalRegionField(const Legion::LogicalRegion& lr,
                                       Legion::FieldID fid,
                                       std::shared_ptr<LogicalRegionField> parent)
  : lr_(lr), fid_(fid), parent_(std::move(parent))
{
}

int32_t LogicalRegionField::dim() const { return lr_.get_dim(); }

const LogicalRegionField& LogicalRegionField::get_root() const
{
  return parent_ != nullptr ? parent_->get_root() : *this;
}

Domain LogicalRegionField::domain() const
{
  return Runtime::get_runtime()->get_index_space_domain(lr_.get_index_space());
}

std::shared_ptr<LogicalRegionField> LogicalRegionField::get_child(const Tiling* tiling,
                                                                  const Shape& color,
                                                                  bool complete)
{
  auto legion_partition = get_legion_partition(tiling, complete);
  auto color_point      = to_domain_point(color);
  return std::make_shared<LogicalRegionField>(
    Runtime::get_runtime()->get_subregion(legion_partition, color_point), fid_, shared_from_this());
}

Legion::LogicalPartition LogicalRegionField::get_legion_partition(const Partition* partition,
                                                                  bool complete)
{
  return partition->construct(lr_, complete);
}

}  // namespace legate::detail
