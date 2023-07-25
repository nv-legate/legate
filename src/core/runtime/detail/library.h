/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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

class Library {
 private:
  class ResourceIdScope {
   public:
    ResourceIdScope() = default;
    ResourceIdScope(int64_t base, int64_t size) : base_(base), size_(size) {}

   public:
    ResourceIdScope(const ResourceIdScope&) = default;

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
  Library(const std::string& library_name,
          const ResourceConfig& config,
          std::unique_ptr<mapping::Mapper> mapper);

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
  /**
   * @brief Returns the name of a task
   *
   * @param local_task_id Task id
   * @return Name of the task
   */
  const std::string& get_task_name(int64_t local_task_id) const;
  void register_mapper(std::unique_ptr<mapping::Mapper> mapper);

 public:
  void register_task(int64_t local_task_id, std::unique_ptr<TaskInfo> task_info);
  const TaskInfo* find_task(int64_t local_task_id) const noexcept(false);

 private:
  void perform_callback(Legion::RegistrationWithArgsCallbackFnptr callback,
                        Legion::UntypedBuffer buffer);

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
  std::unordered_map<int64_t, std::unique_ptr<TaskInfo>> tasks_;
};

}  // namespace legate::detail
