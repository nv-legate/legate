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

#include <memory>
#include <unordered_map>

#include "core/mapping/mapping.h"
#include "core/runtime/resource.h"
#include "core/task/task_info.h"
#include "core/utilities/typedefs.h"

namespace legate::detail {

class Runtime;

void register_mapper_callback(const Legion::RegistrationCallbackArgs& args);

class Library {
 private:
  class ResourceIdScope {
   public:
    ResourceIdScope() = default;
    ResourceIdScope(int64_t base, int64_t size) : base_(base), size_(size) {}

   public:
    int64_t translate(int64_t local_resource_id) const { return base_ + local_resource_id; }
    int64_t invert(int64_t resource_id) const
    {
      assert(in_scope(resource_id));
      return resource_id - base_;
    }
    int64_t generate_id()
    {
      if (next_ == size_) throw std::overflow_error("The scope ran out of IDs");
      return next_++;
    }

   public:
    bool valid() const { return base_ != -1; }
    bool in_scope(int64_t resource_id) const
    {
      return base_ <= resource_id && resource_id < base_ + size_;
    }
    int64_t size() const { return size_; }

   private:
    int64_t base_{-1};
    int64_t size_{-1};
    int64_t next_{0};
  };

 private:
  friend class Runtime;
  Library(const std::string& library_name, const ResourceConfig& config);

 public:
  Library(const Library&) = delete;
  Library(Library&&)      = delete;

 public:
  const std::string& get_library_name() const;

 public:
  Legion::TaskID get_task_id(int64_t local_task_id) const;
  Legion::MapperID get_mapper_id() const { return mapper_id_; }
  Legion::ReductionOpID get_reduction_op_id(int64_t local_redop_id) const;
  Legion::ProjectionID get_projection_id(int64_t local_proj_id) const;
  Legion::ShardingID get_sharding_id(int64_t local_shard_id) const;

 public:
  int64_t get_local_task_id(Legion::TaskID task_id) const;
  int64_t get_local_reduction_op_id(Legion::ReductionOpID redop_id) const;
  int64_t get_local_projection_id(Legion::ProjectionID proj_id) const;
  int64_t get_local_sharding_id(Legion::ShardingID shard_id) const;

 public:
  bool valid_task_id(Legion::TaskID task_id) const;
  bool valid_reduction_op_id(Legion::ReductionOpID redop_id) const;
  bool valid_projection_id(Legion::ProjectionID proj_id) const;
  bool valid_sharding_id(Legion::ShardingID shard_id) const;

 public:
  int64_t get_new_task_id() { return task_scope_.generate_id(); }

 public:
  const std::string& get_task_name(int64_t local_task_id) const;
  void register_mapper(std::unique_ptr<mapping::Mapper> mapper, bool in_callback);
  Legion::Mapping::Mapper* get_legion_mapper() const { return legion_mapper_; }

 public:
  void register_task(int64_t local_task_id, std::unique_ptr<TaskInfo> task_info);
  const TaskInfo* find_task(int64_t local_task_id) const;

 private:
  Legion::Runtime* runtime_;
  const std::string library_name_;
  ResourceIdScope task_scope_;
  ResourceIdScope redop_scope_;
  ResourceIdScope proj_scope_;
  ResourceIdScope shard_scope_;

 private:
  Legion::MapperID mapper_id_;
  std::unique_ptr<mapping::Mapper> mapper_;
  Legion::Mapping::Mapper* legion_mapper_;
  std::unordered_map<int64_t, std::unique_ptr<TaskInfo>> tasks_;
};

}  // namespace legate::detail
