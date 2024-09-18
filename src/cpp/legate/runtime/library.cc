/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/runtime/library.h"

#include "legate/runtime/detail/library.h"

namespace legate {

std::string_view Library::get_library_name() const
{
  return impl()->get_library_name().as_string_view();
}

GlobalTaskID Library::get_task_id(LocalTaskID local_task_id) const
{
  return impl()->get_task_id(local_task_id);
}

GlobalRedopID Library::get_reduction_op_id(LocalRedopID local_redop_id) const
{
  return impl()->get_reduction_op_id(local_redop_id);
}

Legion::ProjectionID Library::get_projection_id(std::int64_t local_proj_id) const
{
  return impl()->get_projection_id(local_proj_id);
}

Legion::ShardingID Library::get_sharding_id(std::int64_t local_shard_id) const
{
  return impl()->get_sharding_id(local_shard_id);
}

LocalTaskID Library::get_local_task_id(GlobalTaskID task_id) const
{
  return impl()->get_local_task_id(task_id);
}

LocalRedopID Library::get_local_reduction_op_id(GlobalRedopID redop_id) const
{
  return impl()->get_local_reduction_op_id(redop_id);
}

std::int64_t Library::get_local_projection_id(Legion::ProjectionID proj_id) const
{
  return impl()->get_local_projection_id(proj_id);
}

std::int64_t Library::get_local_sharding_id(Legion::ShardingID shard_id) const
{
  return impl()->get_local_sharding_id(shard_id);
}

bool Library::valid_task_id(GlobalTaskID task_id) const { return impl()->valid_task_id(task_id); }

bool Library::valid_reduction_op_id(GlobalRedopID redop_id) const
{
  return impl()->valid_reduction_op_id(redop_id);
}

bool Library::valid_projection_id(Legion::ProjectionID proj_id) const
{
  return impl()->valid_projection_id(proj_id);
}

bool Library::valid_sharding_id(Legion::ShardingID shard_id) const
{
  return impl()->valid_sharding_id(shard_id);
}

LocalTaskID Library::get_new_task_id() { return impl()->get_new_task_id(); }

std::string_view Library::get_task_name(LocalTaskID local_task_id) const
{
  return impl()->get_task_name(local_task_id);
}

Scalar Library::get_tunable(std::int64_t tunable_id, const Type& type)
{
  return Scalar{impl()->get_tunable(tunable_id, type.impl())};
}

void Library::register_task(LocalTaskID local_task_id, std::unique_ptr<TaskInfo> task_info)
{
  impl()->register_task(local_task_id, std::move(task_info));
}

const std::map<VariantCode, VariantOptions>& Library::get_default_variant_options() const
{
  return impl()->get_default_variant_options();
}

const TaskInfo* Library::find_task(LocalTaskID local_task_id) const
{
  return impl()->find_task(local_task_id);
}

void Library::perform_callback_(Legion::RegistrationWithArgsCallbackFnptr callback,
                                const Legion::UntypedBuffer& buffer)
{
  Legion::Runtime::perform_registration_callback(callback, buffer, false /*global*/);
}

}  // namespace legate
