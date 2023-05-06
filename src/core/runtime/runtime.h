/* Copyright 2021-2022 NVIDIA Corporation
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

#include <memory>

#include "core/data/scalar.h"
#include "core/data/shape.h"
#include "core/data/store.h"
#include "core/legate_c.h"
#include "core/runtime/resource.h"
#include "core/task/exception.h"
#include "core/utilities/typedefs.h"

/** @defgroup runtime Runtime and library contexts
 */

namespace legate {

class LibraryContext;

namespace mapping {

class Mapper;

}  // namespace mapping

extern uint32_t extract_env(const char* env_name,
                            const uint32_t default_value,
                            const uint32_t test_value);

/**
 * @ingroup runtime
 * @brief A utility class that collects static members shared by all Legate libraries
 */
struct Core {
 public:
  static void parse_config(void);
  static void shutdown(void);
  static void show_progress(const Legion::Task* task,
                            Legion::Context ctx,
                            Legion::Runtime* runtime);
  static void report_unexpected_exception(const Legion::Task* task, const legate::TaskException& e);
  static void retrieve_tunable(Legion::Context legion_context,
                               Legion::Runtime* legion_runtime,
                               LibraryContext* context);

 public:
  /**
   * @brief Type signature for registration callbacks
   */
  using RegistrationCallback = void (*)();

  /**
   * @brief Performs a registration callback. Libraries must perform
   * registration of tasks and other components through this function.
   *
   * @tparam CALLBACK Registration callback to perform
   */
  template <RegistrationCallback CALLBACK>
  static void perform_registration();

 public:
  // Configuration settings
  static bool show_progress_requested;
  static bool use_empty_task;
  static bool synchronize_stream_view;
  static bool log_mapping_decisions;
  static bool has_socket_mem;
  static bool standalone;
  static LegateMainFnPtr main_fn;
};

class AutoTask;
class FieldManager;
class LogicalRegionField;
class LogicalStore;
class ManualTask;
class Operation;
class PartitioningFunctor;
class RegionManager;
class ResourceConfig;
class Runtime;
class Tiling;

class PartitionManager {
 public:
  PartitionManager(Runtime* runtime, const LibraryContext* context);

 public:
  Shape compute_launch_shape(const Shape& shape);
  Shape compute_tile_shape(const Shape& extents, const Shape& launch_shape);

 public:
  Legion::IndexPartition find_index_partition(const Legion::IndexSpace& index_space,
                                              const Tiling& tiling) const;
  void record_index_partition(const Legion::IndexSpace& index_space,
                              const Tiling& tiling,
                              const Legion::IndexPartition& index_partition);

 private:
  int32_t num_pieces_;
  int64_t min_shard_volume_;
  std::vector<size_t> piece_factors_;

 private:
  using TilingCacheKey = std::pair<Legion::IndexSpace, Tiling>;
  std::map<TilingCacheKey, Legion::IndexPartition> tiling_cache_;
};

class Runtime {
 public:
  Runtime(Legion::Runtime* legion_runtime);
  ~Runtime();

 public:
  friend void initialize(int32_t argc, char** argv);
  friend int32_t start(int32_t argc, char** argv);

 public:
  LibraryContext* find_library(const std::string& library_name, bool can_fail = false) const;
  LibraryContext* create_library(const std::string& library_name,
                                 const ResourceConfig& config            = ResourceConfig{},
                                 std::unique_ptr<mapping::Mapper> mapper = nullptr);

 public:
  uint32_t get_type_uid();
  void record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id);
  int32_t find_reduction_operator(int32_t type_uid, int32_t op_kind) const;

 public:
  void enter_callback();
  void exit_callback();
  bool is_in_callback() const;

 public:
  void post_startup_initialization(Legion::Context legion_context);

 public:
  template <typename T>
  T get_tunable(Legion::MapperID mapper_id, int64_t tunable_id);

 public:
  std::unique_ptr<AutoTask> create_task(LibraryContext* library, int64_t task_id);
  std::unique_ptr<ManualTask> create_task(LibraryContext* library,
                                          int64_t task_id,
                                          const Shape& launch_shape);
  void submit(std::unique_ptr<Operation> op);

 public:
  LogicalStore create_store(Type::Code code, int32_t dim = 1);
  LogicalStore create_store(std::vector<size_t> extents,
                            Type::Code code,
                            bool optimize_scalar = false);
  LogicalStore create_store(const Scalar& scalar);
  uint64_t get_unique_store_id();

 public:
  std::shared_ptr<LogicalRegionField> create_region_field(const Shape& extents, Type::Code code);
  std::shared_ptr<LogicalRegionField> import_region_field(Legion::LogicalRegion region,
                                                          Legion::FieldID field_id,
                                                          Type::Code code);
  RegionField map_region_field(LibraryContext* context, const LogicalRegionField* region_field);
  void unmap_physical_region(Legion::PhysicalRegion pr);

 public:
  RegionManager* find_or_create_region_manager(const Legion::Domain& shape);
  FieldManager* find_or_create_field_manager(const Legion::Domain& shape, Type::Code code);
  PartitionManager* partition_manager() const;

 public:
  Legion::IndexSpace find_or_create_index_space(const Legion::Domain& shape);
  Legion::IndexPartition create_restricted_partition(const Legion::IndexSpace& index_space,
                                                     const Legion::IndexSpace& color_space,
                                                     Legion::PartitionKind kind,
                                                     const Legion::DomainTransform& transform,
                                                     const Legion::Domain& extent);
  Legion::FieldSpace create_field_space();
  Legion::LogicalRegion create_region(const Legion::IndexSpace& index_space,
                                      const Legion::FieldSpace& field_space);
  Legion::LogicalPartition create_logical_partition(const Legion::LogicalRegion& logical_region,
                                                    const Legion::IndexPartition& index_partition);
  Legion::Future create_future(const void* data, size_t datalen) const;
  Legion::FieldID allocate_field(const Legion::FieldSpace& field_space, size_t field_size);
  Legion::Domain get_index_space_domain(const Legion::IndexSpace& index_space) const;

 public:
  Legion::Future dispatch(Legion::TaskLauncher* launcher,
                          std::vector<Legion::OutputRequirement>* output_requirements = nullptr);
  Legion::FutureMap dispatch(Legion::IndexTaskLauncher* launcher,
                             std::vector<Legion::OutputRequirement>* output_requirements = nullptr);

 public:
  Legion::Future extract_scalar(const Legion::Future& result, uint32_t idx) const;
  Legion::FutureMap extract_scalar(const Legion::FutureMap& result,
                                   uint32_t idx,
                                   const Legion::Domain& launch_domain) const;
  Legion::Future reduce_future_map(const Legion::FutureMap& future_map, int32_t reduction_op) const;

 public:
  void issue_execution_fence(bool block = false);

 public:
  Legion::ProjectionID get_projection(int32_t src_ndim, const proj::SymbolicPoint& point);
  Legion::ProjectionID get_delinearizing_projection();

 private:
  void schedule(std::vector<std::unique_ptr<Operation>> operations);

 public:
  static void initialize(int32_t argc, char** argv);
  static int32_t start(int32_t argc, char** argv);

 public:
  static Runtime* get_runtime();
  static void create_runtime(Legion::Runtime* legion_runtime);

 private:
  static Runtime* runtime_;

 private:
  Legion::Runtime* legion_runtime_;
  Legion::Context legion_context_{nullptr};
  LibraryContext* core_context_{nullptr};

 private:
  std::map<std::pair<Legion::Domain, Type::Code>, FieldManager*> field_managers_;
  std::map<Legion::Domain, RegionManager*> region_managers_;
  PartitionManager* partition_manager_{nullptr};

 private:
  std::map<Legion::Domain, Legion::IndexSpace> index_spaces_;

 private:
  using ProjectionDesc = std::pair<int32_t, proj::SymbolicPoint>;
  int64_t next_projection_id_{LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID};
  std::map<ProjectionDesc, Legion::ProjectionID> registered_projections_;

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  size_t window_size_{1};
  uint64_t next_unique_id_{0};

 private:
  using RegionFieldID = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::map<RegionFieldID, Legion::PhysicalRegion> inline_mapped_;
  uint64_t next_store_id_{1};

 private:
  bool in_callback_{false};

 private:
  std::map<std::string, LibraryContext*> libraries_{};

 private:
  uint32_t next_type_uid_;
  std::map<std::pair<int32_t, int32_t>, int32_t> reduction_ops_{};
};

void initialize(int32_t argc, char** argv);

void set_main_function(LegateMainFnPtr p_main);

int32_t start(int32_t argc, char** argv);

}  // namespace legate

#include "core/runtime/runtime.inl"
