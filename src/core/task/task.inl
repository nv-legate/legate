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

#pragma once

// Useful for IDEs
#include "core/task/task.h"

namespace legate::detail {

std::string generate_task_name(const std::type_info&);

void task_wrapper(
  VariantImpl, const std::string&, const void*, size_t, const void*, size_t, Legion::Processor);

};  // namespace legate::detail

namespace legate {

template <typename T>
template <VariantImpl VARIANT_IMPL>
/*static*/ void LegateTask<T>::legate_task_wrapper(
  const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
{
  detail::task_wrapper(VARIANT_IMPL, task_name(), args, arglen, userdata, userlen, p);
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  auto task_info = create_task_info(all_options);
  T::Registrar::get_registrar().record_task(T::TASK_ID, std::move(task_info));
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  Library library, const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  register_variants(library, T::TASK_ID, all_options);
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  Library library, int64_t task_id, const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  auto task_info = create_task_info(all_options);
  library.register_task(task_id, std::move(task_info));
}

template <typename T>
/*static*/ std::unique_ptr<TaskInfo> LegateTask<T>::create_task_info(
  const std::map<LegateVariantCode, VariantOptions>& all_options)
{
  auto task_info = std::make_unique<TaskInfo>(task_name());
  detail::VariantHelper<T, detail::CPUVariant>::record(task_info.get(), all_options);
  detail::VariantHelper<T, detail::OMPVariant>::record(task_info.get(), all_options);
  detail::VariantHelper<T, detail::GPUVariant>::record(task_info.get(), all_options);
  return task_info;
}

template <typename T>
/*static*/ const std::string& LegateTask<T>::task_name()
{
  static const std::string result = detail::generate_task_name(typeid(T));
  return result;
}

}  // namespace legate
