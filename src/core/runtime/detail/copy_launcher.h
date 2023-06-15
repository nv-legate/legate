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
#include <optional>

#include "core/mapping/machine.h"

namespace legate {
class BufferBuilder;
class LibraryContext;
class Scalar;
}  // namespace legate

namespace legate::mapping {
class MachineDesc;
}  // namespace legate::mapping

namespace legate::detail {

class LogicalStore;
class ProjectionInfo;
class OutputRequirementAnalyzer;
class CopyArg;
class RequirementAnalyzer;

class CopyLauncher {
 public:
  CopyLauncher(LibraryContext* library,
               const mapping::MachineDesc& machine,
               bool source_indirect_out_of_range,
               bool target_indirect_out_of_range,
               int64_t tag = 0);
  ~CopyLauncher();

 public:
  int64_t legion_mapper_id() const;

 public:
  void add_input(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_output(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_inout(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_reduction(detail::LogicalStore* store,
                     std::unique_ptr<ProjectionInfo> proj_info,
                     bool read_write);
  void add_source_indirect(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);
  void add_target_indirect(detail::LogicalStore* store, std::unique_ptr<ProjectionInfo> proj_info);

 private:
  void add_store(std::vector<CopyArg*>& args,
                 detail::LogicalStore* store,
                 std::unique_ptr<ProjectionInfo> proj_info,
                 Legion::PrivilegeMode privilege);

 public:
  void execute(const Legion::Domain& launch_domain);
  void execute_single();

 private:
  void pack_args();
  void pack_sharding_functor_id();
  std::unique_ptr<Legion::IndexCopyLauncher> build_index_copy(const Legion::Domain& launch_domain);
  std::unique_ptr<Legion::CopyLauncher> build_single_copy();
  template <class Launcher>
  void populate_copy(Launcher* launcher);

 private:
  LibraryContext* library_;
  int64_t tag_;
  mapping::MachineDesc machine_;
  Legion::ProjectionID key_proj_id_{0};

 private:
  BufferBuilder* mapper_arg_;
  std::vector<CopyArg*> inputs_{};
  std::vector<CopyArg*> outputs_{};
  std::vector<CopyArg*> source_indirect_{};
  std::vector<CopyArg*> target_indirect_{};

 private:
  bool source_indirect_out_of_range_;
  bool target_indirect_out_of_range_;
};

}  // namespace legate::detail
