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

#include "core/runtime/library.h"

#include "core/mapping/mapping.h"
#include "core/runtime/detail/library.h"

namespace legate {

const std::string& Library::get_library_name() const { return impl_->get_library_name(); }

Legion::TaskID Library::get_task_id(int64_t local_task_id) const
{
  return impl_->get_task_id(local_task_id);
}

Legion::MapperID Library::get_mapper_id() const { return impl_->get_mapper_id(); }

Legion::ReductionOpID Library::get_reduction_op_id(int64_t local_redop_id) const
{
  return impl_->get_reduction_op_id(local_redop_id);
}

Legion::ProjectionID Library::get_projection_id(int64_t local_proj_id) const
{
  return impl_->get_projection_id(local_proj_id);
}

Legion::ShardingID Library::get_sharding_id(int64_t local_shard_id) const
{
  return impl_->get_sharding_id(local_shard_id);
}

int64_t Library::get_local_task_id(Legion::TaskID task_id) const
{
  return impl_->get_local_task_id(task_id);
}

int64_t Library::get_local_reduction_op_id(Legion::ReductionOpID redop_id) const
{
  return impl_->get_local_reduction_op_id(redop_id);
}

int64_t Library::get_local_projection_id(Legion::ProjectionID proj_id) const
{
  return impl_->get_local_projection_id(proj_id);
}

int64_t Library::get_local_sharding_id(Legion::ShardingID shard_id) const
{
  return impl_->get_local_sharding_id(shard_id);
}

bool Library::valid_task_id(Legion::TaskID task_id) const { return impl_->valid_task_id(task_id); }

bool Library::valid_reduction_op_id(Legion::ReductionOpID redop_id) const
{
  return impl_->valid_reduction_op_id(redop_id);
}

bool Library::valid_projection_id(Legion::ProjectionID proj_id) const
{
  return impl_->valid_projection_id(proj_id);
}

bool Library::valid_sharding_id(Legion::ShardingID shard_id) const
{
  return impl_->valid_sharding_id(shard_id);
}

int64_t Library::get_new_task_id() { return impl_->get_new_task_id(); }

const std::string& Library::get_task_name(int64_t local_task_id) const
{
  return impl_->get_task_name(local_task_id);
}

void Library::register_mapper(std::unique_ptr<mapping::Mapper> mapper)
{
  impl_->register_mapper(std::move(mapper), false /*in_callback*/);
}

void Library::register_task(int64_t local_task_id, std::unique_ptr<TaskInfo> task_info)
{
  impl_->register_task(local_task_id, std::move(task_info));
}

Library::Library(detail::Library* impl) : impl_(impl) {}

bool Library::operator==(const Library& other) const { return impl_ == other.impl_; }

bool Library::operator!=(const Library& other) const { return impl_ != other.impl_; }

void Library::perform_callback(Legion::RegistrationWithArgsCallbackFnptr callback,
                               Legion::UntypedBuffer buffer)
{
  Legion::Runtime::perform_registration_callback(callback, buffer, true /*global*/);
}

}  // namespace legate
