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

#include "core/runtime/detail/library.h"

#include <sstream>
#include <stdexcept>

namespace legate::detail {

inline Library::ResourceIdScope::ResourceIdScope(int64_t base, int64_t size, int64_t dyn_size)
  : base_{base}, size_{size}, next_{size - dyn_size}
{
  if (LegateDefined(LEGATE_USE_DEBUG) && (dyn_size > this->size())) {
    std::stringstream ss;

    ss << "Number of dynamic resource IDs " << dyn_size << " > total number of IDs "
       << this->size();
    throw std::out_of_range{std::move(ss).str()};
  }
}

inline int64_t Library::ResourceIdScope::translate(int64_t local_resource_id) const
{
  return base_ + local_resource_id;
}

inline int64_t Library::ResourceIdScope::invert(int64_t resource_id) const
{
  LegateCheck(in_scope(resource_id));
  return resource_id - base_;
}

inline int64_t Library::ResourceIdScope::generate_id()
{
  if (next_ == size_) {
    throw std::overflow_error{"The scope ran out of IDs"};
  }
  return next_++;
}

inline bool Library::ResourceIdScope::valid() const { return base_ != -1; }

inline bool Library::ResourceIdScope::in_scope(int64_t resource_id) const
{
  return base_ <= resource_id && resource_id < base_ + size_;
}

inline int64_t Library::ResourceIdScope::size() const { return size_; }

// ==========================================================================================

inline const std::string& Library::get_library_name() const { return library_name_; }

inline Legion::MapperID Library::get_mapper_id() const { return mapper_id_; }

inline int64_t Library::get_new_task_id() { return task_scope_.generate_id(); }

inline Legion::Mapping::Mapper* Library::get_legion_mapper() const { return legion_mapper_; }

}  // namespace legate::detail
