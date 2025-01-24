/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/mapping/detail/machine.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/task/detail/task_context.h>
#include <legate/task/detail/task_return.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace legate::detail {

class LegionTaskContext final : public TaskContext {
 public:
  LegionTaskContext(const Legion::Task* legion_task,
                    VariantCode variant_kind,
                    const std::vector<Legion::PhysicalRegion>& regions);

  [[nodiscard]] GlobalTaskID task_id() const noexcept override;
  [[nodiscard]] bool is_single_task() const noexcept override;
  [[nodiscard]] const DomainPoint& get_task_index() const noexcept override;
  [[nodiscard]] const Domain& get_launch_domain() const noexcept override;
  [[nodiscard]] std::string_view get_provenance() const override;
  [[nodiscard]] const mapping::detail::Machine& machine() const noexcept override;

  [[nodiscard]] TaskReturn pack_return_values(const std::optional<ReturnedException>& exn) const;

 private:
  const Legion::Task* task_{};
  mapping::detail::Machine machine_{};

  LegionTaskContext(const Legion::Task* legion_task,
                    VariantCode variant_kind,
                    const std::vector<Legion::PhysicalRegion>& regions,
                    mapping::detail::Machine&& machine);

  [[nodiscard]] const Legion::Task& legion_task_() const noexcept;
  [[nodiscard]] std::vector<ReturnValue> get_return_values_() const;
};

void legion_task_body(VariantImpl variant_impl,
                      VariantCode variant_kind,
                      std::optional<std::string_view> task_name,
                      const void* args,
                      std::size_t arglen,
                      Processor p);

}  // namespace legate::detail
