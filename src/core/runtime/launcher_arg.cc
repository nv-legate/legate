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
  buffer.pack<bool>(scalar_.is_tuple());
  buffer.pack<int32_t>(scalar_.code());
  buffer.pack_buffer(scalar_.ptr(), scalar_.size());
}

RegionFieldArg::RegionFieldArg(RequirementAnalyzer* analyzer,
                               LogicalStore store,
                               Legion::FieldID field_id,
                               Legion::PrivilegeMode privilege,
                               const ProjectionInfo* proj_info)
  : analyzer_(analyzer),
    store_(std::move(store)),
    region_(store_.get_storage()->region()),
    field_id_(field_id),
    privilege_(privilege),
    proj_info_(proj_info)
{
}

void RegionFieldArg::pack(BufferBuilder& buffer) const
{
  store_.pack(buffer);
  buffer.pack<int32_t>(proj_info_->redop);
  buffer.pack<int32_t>(region_.get_dim());
  buffer.pack<uint32_t>(analyzer_->get_requirement_index(region_, privilege_, proj_info_));
  buffer.pack<uint32_t>(field_id_);
}

FutureStoreArg::FutureStoreArg(LogicalStore store,
                               bool read_only,
                               bool has_storage,
                               Legion::ReductionOpID redop)
  : store_(std::move(store)), read_only_(read_only), has_storage_(has_storage), redop_(redop)
{
}

struct datalen_fn {
  template <LegateTypeCode CODE>
  size_t operator()()
  {
    return sizeof(legate_type_of<CODE>);
  }
};

void FutureStoreArg::pack(BufferBuilder& buffer) const
{
  store_.pack(buffer);

  buffer.pack<int32_t>(redop_);
  buffer.pack<bool>(read_only_);
  buffer.pack<bool>(has_storage_);
  buffer.pack<int32_t>(type_dispatch(store_.code(), datalen_fn{}));
  buffer.pack<size_t>(store_.extents());
}

}  // namespace legate
