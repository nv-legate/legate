/* Copyright 2022 NVIDIA Corporation
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

#include "core/runtime/launcher_arg.h"
#include "core/data/logical_region_field.h"
#include "core/runtime/launcher.h"
#include "core/runtime/req_analyzer.h"

namespace legate {

void UntypedScalarArg::pack(BufferBuilder& buffer) const
{
  // FIXME: Need to catch up the type system change
  // buffer.pack<bool>(scalar_.is_tuple());
  buffer.pack<int32_t>(static_cast<int32_t>(scalar_.type().code));
  buffer.pack_buffer(scalar_.ptr(), scalar_.size());
}

RegionFieldArg::RegionFieldArg(RequirementAnalyzer* analyzer,
                               detail::LogicalStore* store,
                               Legion::FieldID field_id,
                               Legion::PrivilegeMode privilege,
                               const ProjectionInfo* proj_info)
  : analyzer_(analyzer),
    store_(store),
    region_(store_->get_region_field()->region()),
    field_id_(field_id),
    privilege_(privilege),
    proj_info_(proj_info)
{
}

void RegionFieldArg::pack(BufferBuilder& buffer) const
{
  store_->pack(buffer);

  buffer.pack<int32_t>(proj_info_->redop);
  buffer.pack<int32_t>(region_.get_dim());
  buffer.pack<uint32_t>(analyzer_->get_requirement_index(region_, privilege_, proj_info_));
  buffer.pack<uint32_t>(field_id_);
}

OutputRegionArg::OutputRegionArg(OutputRequirementAnalyzer* analyzer,
                                 detail::LogicalStore* store,
                                 Legion::FieldSpace field_space,
                                 Legion::FieldID field_id)
  : analyzer_(analyzer), store_(store), field_space_(field_space), field_id_(field_id)
{
}

void OutputRegionArg::pack(BufferBuilder& buffer) const
{
  store_->pack(buffer);

  requirement_index_ = analyzer_->get_requirement_index(field_space_, field_id_);

  buffer.pack<int32_t>(-1);
  buffer.pack<int32_t>(store_->dim());
  buffer.pack<uint32_t>(requirement_index_);
  buffer.pack<uint32_t>(field_id_);
}

FutureStoreArg::FutureStoreArg(detail::LogicalStore* store,
                               bool read_only,
                               bool has_storage,
                               Legion::ReductionOpID redop)
  : store_(store), read_only_(read_only), has_storage_(has_storage), redop_(redop)
{
}

void FutureStoreArg::pack(BufferBuilder& buffer) const
{
  store_->pack(buffer);

  buffer.pack<int32_t>(redop_);
  buffer.pack<bool>(read_only_);
  buffer.pack<bool>(has_storage_);
  buffer.pack<uint32_t>(static_cast<int32_t>(store_->code()));
  buffer.pack<size_t>(store_->extents().data());
}

}  // namespace legate
