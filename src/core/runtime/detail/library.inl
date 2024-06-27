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

#pragma once

#include "core/runtime/detail/library.h"

#include <fmt/format.h>
#include <stdexcept>

namespace legate::detail {

inline Library::ResourceIdScope::ResourceIdScope(std::int64_t base,
                                                 std::int64_t size,
                                                 std::int64_t dyn_size)
  : base_{base}, size_{size}, next_{size - dyn_size}
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG) && (dyn_size > this->size())) {
    throw std::out_of_range{fmt::format(
      "Number of dynamic resource IDs {} > total number of IDs {}", dyn_size, this->size())};
  }
}

inline std::int64_t Library::ResourceIdScope::translate(std::int64_t local_resource_id) const
{
  if (local_resource_id >= size_) {
    throw std::out_of_range{fmt::format(
      "Maximum local ID is {} but received a local ID {}", size_ - 1, local_resource_id)};
  }
  return base_ + local_resource_id;
}

inline std::int64_t Library::ResourceIdScope::invert(std::int64_t resource_id) const
{
  LEGATE_CHECK(in_scope(resource_id));
  return resource_id - base_;
}

inline std::int64_t Library::ResourceIdScope::generate_id()
{
  if (next_ == size_) {
    throw std::overflow_error{"The scope ran out of IDs"};
  }
  return next_++;
}

inline bool Library::ResourceIdScope::in_scope(std::int64_t resource_id) const
{
  return base_ <= resource_id && resource_id < base_ + size_;
}

inline std::int64_t Library::ResourceIdScope::size() const { return size_; }

// ==========================================================================================

inline std::string_view Library::get_library_name() const { return library_name_; }

inline Legion::MapperID Library::get_mapper_id() const { return mapper_id_; }

inline std::int64_t Library::get_new_task_id() { return task_scope_.generate_id(); }

inline Legion::Mapping::Mapper* Library::get_legion_mapper() const { return legion_mapper_; }

}  // namespace legate::detail
