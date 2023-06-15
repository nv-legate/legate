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

#include "core/mapping/machine.h"

namespace legate {
class BufferBuilder;
class LibraryContext;
}  // namespace legate

namespace legate::mapping {
class MachineDesc;
}  // namespace legate::mapping

namespace legate::detail {

class LogicalStore;
class ProjectionInfo;

class FillLauncher {
 public:
  FillLauncher(LibraryContext* library, const mapping::MachineDesc& machine, int64_t tag = 0);
  ~FillLauncher();

 public:
  void launch(const Legion::Domain& launch_domain,
              LogicalStore* lhs,
              const ProjectionInfo& lhs_proj,
              LogicalStore* value);
  void launch_single(LogicalStore* lhs, const ProjectionInfo& lhs_proj, LogicalStore* value);

 private:
  void pack_mapper_arg(Legion::ProjectionID proj_id);
  std::unique_ptr<Legion::IndexFillLauncher> build_index_fill(const Legion::Domain& launch_domain,
                                                              LogicalStore* lhs,
                                                              const ProjectionInfo& lhs_proj,
                                                              LogicalStore* value);
  std::unique_ptr<Legion::FillLauncher> build_single_fill(LogicalStore* lhs,
                                                          const ProjectionInfo& lhs_proj,
                                                          LogicalStore* value);

 private:
  LibraryContext* library_;
  int64_t tag_;
  mapping::MachineDesc machine_;

 private:
  BufferBuilder* mapper_arg_;
};

}  // namespace legate::detail
