/* Copyright 2021 NVIDIA Corporation
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

#include "legion.h"

#include "core/data/store.h"
#include "core/runtime/context.h"
#include "core/utilities/typedefs.h"

namespace legate {

extern uint32_t extract_env(const char* env_name,
                            const uint32_t default_value,
                            const uint32_t test_value);

class Core {
 public:
  static void parse_config(void);
  static void shutdown(void);

 public:
  // Configuration settings
  static bool show_progress;
  static LegateMainFnPtr main_fn;
#ifdef LEGATE_USE_CUDA
 public:
  static cublasContext* get_cublas(void);
#endif
};

class FieldManager;
class LogicalRegionField;
class LogicalStore;
class Operation;
class PartitioningFunctor;
class RegionManager;
class ResourceConfig;
class Runtime;
class Task;

class PartitionManager {
 public:
  PartitionManager(Runtime* runtime, const LibraryContext* context);

 public:
  std::vector<size_t> compute_launch_shape(const LogicalStore* store);
  std::vector<size_t> compute_tile_shape(const std::vector<size_t>& extents,
                                         const std::vector<size_t>& launch_shape);

 private:
  int32_t num_pieces_;
  int32_t min_shard_volume_;
  std::vector<size_t> piece_factors_;
};

class Runtime {
 public:
 public:
  Runtime(Legion::Runtime* legion_runtime);
  ~Runtime();

 public:
  friend void initialize(int32_t argc, char** argv);
  friend int32_t start(int32_t argc, char** argv);

 public:
  LibraryContext* find_library(const std::string& library_name, bool can_fail = false) const;
  LibraryContext* create_library(const std::string& library_name, const ResourceConfig& config);

 public:
  void post_startup_initialization(Legion::Context legion_context);

 public:
  template <typename T>
  T get_tunable(const LibraryContext* context, int64_t tunable_id, int64_t mapper_id = 0);

 public:
  std::unique_ptr<Task> create_task(LibraryContext* library,
                                    int64_t task_id,
                                    int64_t mapper_id = 0);
  void submit(std::unique_ptr<Operation> op);

 public:
  std::shared_ptr<LogicalStore> create_store(std::vector<size_t> extents, LegateTypeCode code);
  std::shared_ptr<LogicalRegionField> create_region_field(const std::vector<size_t>& extents,
                                                          LegateTypeCode code);
  RegionField map_region_field(LibraryContext* context,
                               std::shared_ptr<LogicalRegionField> region_field);

 public:
  RegionManager* find_or_create_region_manager(const Legion::Domain& shape);
  FieldManager* find_or_create_field_manager(const Legion::Domain& shape, LegateTypeCode code);
  PartitionManager* get_partition_manager();

 public:
  Legion::IndexSpace find_or_create_index_space(const Legion::Domain& shape);
  Legion::IndexPartition create_index_partition(const Legion::IndexSpace& index_space,
                                                const Legion::IndexSpace& color_space,
                                                Legion::PartitionKind kind,
                                                const PartitioningFunctor* functor);
  Legion::FieldSpace create_field_space();
  Legion::LogicalRegion create_region(const Legion::IndexSpace& index_space,
                                      const Legion::FieldSpace& field_space);
  Legion::LogicalPartition create_logical_partition(const Legion::LogicalRegion& logical_region,
                                                    const Legion::IndexPartition& index_partition);
  Legion::FieldID allocate_field(const Legion::FieldSpace& field_space, size_t field_size);
  Legion::Domain get_index_space_domain(const Legion::IndexSpace& index_space) const;

 public:
  std::shared_ptr<LogicalStore> dispatch(Legion::TaskLauncher* launcher);
  std::shared_ptr<LogicalStore> dispatch(Legion::IndexTaskLauncher* launcher);

 private:
  void schedule(std::vector<std::unique_ptr<Operation>> operations);

 public:
  static void initialize(int32_t argc, char** argv);
  static int32_t start(int32_t argc, char** argv);
  static Runtime* get_runtime();
  static void create_runtime(Legion::Runtime* legion_runtime);

 private:
  static Runtime* runtime_;

 private:
  Legion::Runtime* legion_runtime_;
  Legion::Context legion_context_{nullptr};
  LibraryContext* core_context_{nullptr};

 private:
  std::map<std::pair<Legion::Domain, LegateTypeCode>, FieldManager*> field_managers_;
  std::map<Legion::Domain, RegionManager*> region_managers_;
  PartitionManager* partition_manager_{nullptr};

 private:
  std::map<Legion::Domain, Legion::IndexSpace> index_spaces_;

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  size_t window_size_{1};

 private:
  std::map<std::string, LibraryContext*> libraries_;
};

void initialize(int32_t argc, char** argv);

void set_main_function(LegateMainFnPtr p_main);

int32_t start(int32_t argc, char** argv);

}  // namespace legate

#include "core/runtime/runtime.inl"
