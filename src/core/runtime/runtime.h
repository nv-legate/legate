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

class ResourceConfig;
class Runtime;
class Operation;
class Task;
class LibraryContext;
class LogicalRegionField;
class LogicalStore;

class RegionManager {
 public:
  RegionManager(Runtime* runtime, const Legion::Domain& shape);

 private:
  Legion::LogicalRegion active_region() const;
  bool has_space() const;
  void create_region();

 public:
  std::pair<Legion::LogicalRegion, Legion::FieldID> allocate_field(size_t field_size);

 private:
  Runtime* runtime_;
  Legion::Domain shape_;
  std::vector<Legion::LogicalRegion> regions_{};
};

class FieldManager {
 public:
  FieldManager(Runtime* runtime, const Legion::Domain& shape, LegateTypeCode code);

 public:
  std::shared_ptr<LogicalRegionField> allocate_field();

 private:
  Runtime* runtime_;
  Legion::Domain shape_;
  LegateTypeCode code_;
  size_t field_size_;
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
  void set_legion_context(Legion::Context legion_context);

 public:
  std::unique_ptr<Task> create_task(LibraryContext* library,
                                    int64_t task_id,
                                    int64_t mapper_id = 0);
  void submit(std::unique_ptr<Operation> op);

 public:
  std::shared_ptr<LogicalStore> create_store(std::vector<int64_t> extents, LegateTypeCode code);
  std::shared_ptr<LogicalRegionField> create_region_field(const std::vector<int64_t>& extents,
                                                          LegateTypeCode code);
  RegionField map_region_field(LibraryContext* context,
                               std::shared_ptr<LogicalRegionField> region_field);

 public:
  RegionManager* find_or_create_region_manager(const Legion::Domain& shape);
  FieldManager* find_or_create_field_manager(const Legion::Domain& shape, LegateTypeCode code);

 public:
  Legion::IndexSpace find_or_create_index_space(const Legion::Domain& shape);
  Legion::FieldSpace create_field_space();
  Legion::LogicalRegion create_region(const Legion::IndexSpace& index_space,
                                      const Legion::FieldSpace& field_space);
  Legion::FieldID allocate_field(const Legion::FieldSpace& field_space, size_t field_size);
  Legion::Domain get_index_space_domain(const Legion::IndexSpace& index_space) const;

 public:
  std::shared_ptr<LogicalStore> dispatch(Legion::TaskLauncher* launcher);

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

 private:
  std::map<std::pair<Legion::Domain, LegateTypeCode>, FieldManager*> field_managers_;
  std::map<Legion::Domain, RegionManager*> region_managers_;

 private:
  std::map<Legion::Domain, Legion::IndexSpace> index_spaces_;

 private:
  std::map<std::string, LibraryContext*> libraries_;
};

void initialize(int32_t argc, char** argv);

void set_main_function(LegateMainFnPtr p_main);

int32_t start(int32_t argc, char** argv);

}  // namespace legate
