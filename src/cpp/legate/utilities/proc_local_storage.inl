/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/proc_local_storage.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <typeinfo>

namespace legate {

namespace detail {

constexpr std::size_t LOCAL_PROC_BITWIDTH = Realm::ID::FMT_Processor::proc_idx::BITS;
constexpr std::size_t LOCAL_PROC_MASK     = (1 << LOCAL_PROC_BITWIDTH) - 1;

[[nodiscard]] inline std::size_t processor_id()
{
  // Processor IDs are numbered locally in each node and local indices are encoded in the LSBs, so
  // here we mask out the rest to get the rank-local index of the processor
  const auto proc =
    Legion::Runtime::get_runtime()->get_executing_processor(Legion::Runtime::get_context());
  return static_cast<std::size_t>(proc.id & LOCAL_PROC_MASK);
}

[[noreturn]] void throw_invalid_proc_local_storage_access(const std::type_info&);

}  // namespace detail

template <typename T>
bool ProcLocalStorage<T>::has_value() const
{
  return storage_[detail::processor_id()].has_value();
}

template <typename T>
template <typename... Args>
typename ProcLocalStorage<T>::value_type& ProcLocalStorage<T>::emplace(Args&&... args)
{
  return storage_[detail::processor_id()].emplace(std::forward<Args>(args)...);
}

template <typename T>
constexpr T& ProcLocalStorage<T>::get()
{
  auto& entry = storage_[detail::processor_id()];
  if (!entry.has_value()) {
    detail::throw_invalid_proc_local_storage_access(typeid(T));
  }
  return *entry;
}

template <typename T>
constexpr const T& ProcLocalStorage<T>::get() const
{
  auto& entry = storage_[detail::processor_id()];
  if (!entry.has_value()) {
    detail::throw_invalid_proc_local_storage_access(typeid(T));
  }
  return *entry;
}

}  // namespace legate
