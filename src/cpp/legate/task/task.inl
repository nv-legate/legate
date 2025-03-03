/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate_defines.h>

#include <legate/task/registrar.h>
#include <legate/task/task.h>
#include <legate/utilities/compiler.h>

namespace legate {

template <typename T>
/*static*/ void LegateTask<T>::register_variants(std::map<VariantCode, VariantOptions> all_options)
{
  T::Registrar::get_registrar().record_task(
    TaskRegistrar::RecordTaskKey{},
    T::TASK_ID,
    [callsite_options = std::move(all_options)](const Library& lib) {
      return create_task_info_(lib, callsite_options);
    });
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  Library library, const std::map<VariantCode, VariantOptions>& all_options)
{
  register_variants(std::move(library), static_cast<LocalTaskID>(T::TASK_ID), all_options);
}

template <typename T>
/*static*/ void LegateTask<T>::register_variants(
  Library library, LocalTaskID task_id, const std::map<VariantCode, VariantOptions>& all_options)
{
  const auto task_info = create_task_info_(library, all_options);

  library.register_task(task_id, task_info);
}

template <typename T>
/*static*/ TaskInfo LegateTask<T>::create_task_info_(
  const Library& lib, const std::map<VariantCode, VariantOptions>& all_options)
{
  auto task_info = TaskInfo{task_name_().to_string()};

  detail::VariantHelper<T, detail::CPUVariant>::record(lib, all_options, &task_info);
  detail::VariantHelper<T, detail::OMPVariant>::record(lib, all_options, &task_info);
  detail::VariantHelper<T, detail::GPUVariant>::record(lib, all_options, &task_info);
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
