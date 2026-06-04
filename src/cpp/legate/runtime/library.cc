/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/library.h>

#include <legate/runtime/detail/library.h>

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

LocalTaskID Library::get_local_task_id(GlobalTaskID task_id) const
{
  return impl()->get_local_task_id(task_id);
}

LocalRedopID Library::get_local_reduction_op_id(GlobalRedopID redop_id) const
{
  return impl()->get_local_reduction_op_id(redop_id);
}

bool Library::valid_task_id(GlobalTaskID task_id) const { return impl()->valid_task_id(task_id); }

bool Library::valid_reduction_op_id(GlobalRedopID redop_id) const
{
  return impl()->valid_reduction_op_id(redop_id);
}

LocalTaskID Library::get_new_task_id() { return impl()->get_new_task_id(); }

std::string_view Library::get_task_name(LocalTaskID local_task_id) const
{
  return impl()->get_task_name(local_task_id);
}

void Library::register_task(LocalTaskID local_task_id, const TaskInfo& task_info)
{
  impl()->register_task(local_task_id, task_info.impl());
}

const std::map<VariantCode, VariantOptions>& Library::get_default_variant_options() const
{
  return impl()->get_default_variant_options();
}

TaskInfo Library::find_task(LocalTaskID local_task_id) const
{
  return TaskInfo{impl()->find_task(local_task_id)};
}

void Library::perform_callback_(Legion::RegistrationWithArgsCallbackFnptr callback,
                                const Legion::UntypedBuffer& buffer)
{
  detail::Library::perform_callback(callback, buffer);
}

}  // namespace legate
