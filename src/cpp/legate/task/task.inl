/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/task/registrar.h>
#include <legate/task/task.h>
#include <legate/task/task_config.h>
#include <legate/task/variant_helper.h>
#include <legate/utilities/compiler.h>
#include <legate/utilities/detail/type_traits.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace legate {

namespace legate_task_detail {

template <typename U>
using has_task_config = decltype(U::TASK_CONFIG);

template <typename U>
using has_task_signature = decltype(U::TASK_SIGNATURE);

template <typename U>
using has_task_id = decltype(U::TASK_ID);

}  // namespace legate_task_detail

template <typename T>
/*static*/ void LegateTask<T>::register_variants(std::map<VariantCode, VariantOptions> all_options)
{
  T::Registrar::get_registrar().record_registration_function(
    {}, [callsite_options = std::move(all_options)](const Library& lib) {
      return T::register_variants(lib, callsite_options);
    });
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  Library library, const std::map<VariantCode, VariantOptions>& all_options)
{
  static_assert(detail::is_detected_v<legate_task_detail::has_task_config, T>,
                "Task must define a \"static const legate::TaskConfig TASK_CONFIG\" member");
  static_assert(std::is_same_v<decltype(T::TASK_CONFIG)&, const TaskConfig&>,
                "Incompatible type for TASK_CONFIG. Must be \"static const legate::TaskConfig "
                "TASK_CONFIG\"");
  static_assert(!detail::is_detected_v<legate_task_detail::has_task_signature, T>,
                "TASK_SIGNATURE is deprecated. Please use TASK_CONFIG instead");
  static_assert(!detail::is_detected_v<legate_task_detail::has_task_id, T>,
                "TASK_ID is deprecated. Please use TASK_CONFIG instead");

  register_variants(library, T::TASK_CONFIG, all_options);
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  Library library, LocalTaskID task_id, const std::map<VariantCode, VariantOptions>& all_options)
{
  register_variants(library, TaskConfig{task_id}, all_options);
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  Library library,
  const TaskConfig& task_config,
  const std::map<VariantCode, VariantOptions>& all_options)
{
  const auto task_info = create_task_info_(library, task_config, all_options);

  library.register_task(task_config.task_id(), task_info);
}

template <typename T>
/*static*/ TaskInfo LegateTask<T>::create_task_info_(
  const Library& lib,
  const TaskConfig& task_config,
  const std::map<VariantCode, VariantOptions>& all_options)
{
  auto task_info = TaskInfo{task_name_().to_string()};

  detail::VariantHelper<T, detail::CPUVariant>::record(lib, task_config, all_options, &task_info);
  detail::VariantHelper<T, detail::OMPVariant>::record(lib, task_config, all_options, &task_info);
  detail::VariantHelper<T, detail::GPUVariant>::record(lib, task_config, all_options, &task_info);
  return task_info;
}

template <typename T>
/*static*/ detail::ZStringView LegateTask<T>::task_name_()
{
  static const std::string result = detail::demangle_type(typeid(T));
  return result;
}

template <typename T>
template <VariantImpl variant_fn, VariantCode variant_kind>
/*static*/ void LegateTask<T>::task_wrapper_(const void* args,
                                             std::size_t arglen,
                                             const void* userdata,
                                             std::size_t userlen,
                                             Legion::Processor p)
{
  detail::task_wrapper(variant_fn,
                       variant_kind,
                       task_name_().as_string_view(),
                       args,
                       arglen,
                       userdata,
                       userlen,
                       std::move(p));
}

}  // namespace legate
