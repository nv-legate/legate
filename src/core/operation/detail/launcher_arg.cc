/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/operation/detail/launcher_arg.h"

#include "core/data/detail/logical_region_field.h"
#include "core/operation/detail/req_analyzer.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

void UntypedScalarArg::pack(BufferBuilder& buffer) const { scalar_.pack(buffer); }

RegionFieldArg::RegionFieldArg(RequirementAnalyzer* analyzer,
                               LogicalStore* store,
                               Legion::FieldID field_id,
                               Legion::PrivilegeMode privilege,
                               std::unique_ptr<ProjectionInfo> proj_info)
  : analyzer_(analyzer),
    store_(store),
    region_(store_->get_region_field()->region()),
    field_id_(field_id),
    privilege_(privilege),
    proj_info_(std::move(proj_info))
{
}

void RegionFieldArg::pack(BufferBuilder& buffer) const
{
  store_->pack(buffer);

  buffer.pack<int32_t>(proj_info_->redop);
  buffer.pack<int32_t>(region_.get_dim());
  buffer.pack<uint32_t>(analyzer_->get_requirement_index(region_, privilege_, *proj_info_));
  buffer.pack<uint32_t>(field_id_);
}

OutputRegionArg::OutputRegionArg(OutputRequirementAnalyzer* analyzer,
                                 LogicalStore* store,
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

FutureStoreArg::FutureStoreArg(LogicalStore* store,
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
  buffer.pack<uint32_t>(store_->type()->size());
  buffer.pack<size_t>(store_->get_storage()->extents().data());
}

}  // namespace legate::detail
