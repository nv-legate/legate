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

#include "core/data/detail/scalar.h"
#include "core/mapping/mapping.h"
#include "core/runtime/resource.h"
#include "core/task/task_info.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/typedefs.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace legate::detail {

class Runtime;

void register_mapper_callback(const Legion::RegistrationCallbackArgs& args);

class Library {
  class ResourceIdScope {
   public:
    ResourceIdScope() = default;
    ResourceIdScope(std::int64_t base, std::int64_t size, std::int64_t dyn_size = 0);

    [[nodiscard]] std::int64_t translate(std::int64_t local_resource_id) const;
    [[nodiscard]] std::int64_t invert(std::int64_t resource_id) const;
    [[nodiscard]] std::int64_t generate_id();
    [[nodiscard]] bool in_scope(std::int64_t resource_id) const;
    [[nodiscard]] std::int64_t size() const;

   private:
    std::int64_t base_{};
    std::int64_t size_{};
    std::int64_t next_{};
  };

  friend class Runtime;

 public:
  class ConstructKey {
    ConstructKey() = default;

    friend class Runtime;
    friend class Library;
  };

  Library(ConstructKey,
          std::string library_name,
          const ResourceConfig& config,
          std::map<VariantCode, VariantOptions> default_options);

  Library(const Library&) = delete;
  Library(Library&&)      = delete;

  [[nodiscard]] std::string_view get_library_name() const;

  [[nodiscard]] GlobalTaskID get_task_id(LocalTaskID local_task_id) const;
  [[nodiscard]] Legion::MapperID get_mapper_id() const;
  [[nodiscard]] GlobalRedopID get_reduction_op_id(LocalRedopID local_redop_id) const;
  [[nodiscard]] Legion::ProjectionID get_projection_id(std::int64_t local_proj_id) const;
  [[nodiscard]] Legion::ShardingID get_sharding_id(std::int64_t local_shard_id) const;

  [[nodiscard]] LocalTaskID get_local_task_id(GlobalTaskID task_id) const;
  [[nodiscard]] LocalRedopID get_local_reduction_op_id(GlobalRedopID redop_id) const;
  [[nodiscard]] std::int64_t get_local_projection_id(Legion::ProjectionID proj_id) const;
  [[nodiscard]] std::int64_t get_local_sharding_id(Legion::ShardingID shard_id) const;

  [[nodiscard]] bool valid_task_id(GlobalTaskID task_id) const;
  [[nodiscard]] bool valid_reduction_op_id(GlobalRedopID redop_id) const;
  [[nodiscard]] bool valid_projection_id(Legion::ProjectionID proj_id) const;
  [[nodiscard]] bool valid_sharding_id(Legion::ShardingID shard_id) const;

  [[nodiscard]] LocalTaskID get_new_task_id();

  [[nodiscard]] std::string_view get_task_name(LocalTaskID local_task_id) const;
  [[nodiscard]] std::unique_ptr<Scalar> get_tunable(std::int64_t tunable_id,
                                                    InternalSharedPtr<Type> type) const;
  void register_mapper(std::unique_ptr<mapping::Mapper> mapper, bool in_callback);
  [[nodiscard]] Legion::Mapping::Mapper* get_legion_mapper() const;

  void register_task(LocalTaskID local_task_id, std::unique_ptr<TaskInfo> task_info);
  [[nodiscard]] const TaskInfo* find_task(LocalTaskID local_task_id) const;

  [[nodiscard]] const std::map<VariantCode, VariantOptions>& get_default_variant_options() const;

 private:
  Legion::Runtime* runtime_{};
  std::string library_name_{};
  ResourceIdScope task_scope_{};
  ResourceIdScope redop_scope_{};
  ResourceIdScope proj_scope_{};
  ResourceIdScope shard_scope_{};

  Legion::MapperID mapper_id_{};
  std::unique_ptr<mapping::Mapper> mapper_{};
  Legion::Mapping::Mapper* legion_mapper_{};
  std::unordered_map<LocalTaskID, std::unique_ptr<TaskInfo>> tasks_{};
  std::map<VariantCode, VariantOptions> default_options_{};
};

}  // namespace legate::detail

#include "core/runtime/detail/library.inl"
