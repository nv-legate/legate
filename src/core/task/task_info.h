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

#include "core/task/variant_options.h"
#include "core/utilities/typedefs.h"

#include <memory>

namespace legate {

struct VariantInfo {
  VariantImpl body;
  Legion::CodeDescriptor code_desc;
  VariantOptions options;
};

class TaskInfo {
 public:
  TaskInfo(std::string task_name);
  ~TaskInfo();

  [[nodiscard]] const std::string& name() const;

  void add_variant(LegateVariantCode vid,
                   VariantImpl body,
                   const Legion::CodeDescriptor& code_desc,
                   const VariantOptions& options);
  [[nodiscard]] const VariantInfo& find_variant(LegateVariantCode vid) const;
  [[nodiscard]] bool has_variant(LegateVariantCode vid) const;

  void register_task(Legion::TaskID task_id);

  TaskInfo(const TaskInfo&)            = delete;
  TaskInfo& operator=(const TaskInfo&) = delete;
  TaskInfo(TaskInfo&&)                 = delete;
  TaskInfo& operator=(TaskInfo&&)      = delete;

 private:
  friend std::ostream& operator<<(std::ostream& os, const TaskInfo& info);

  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace legate
