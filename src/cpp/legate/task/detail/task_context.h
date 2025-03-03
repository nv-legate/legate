/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/communicator.h>
#include <legate/cuda/detail/cuda_driver_types.h>
#include <legate/data/detail/physical_array.h>
#include <legate/data/scalar.h>
#include <legate/mapping/detail/machine.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <optional>
#include <string_view>
#include <vector>

namespace legate::detail {

class TaskContext {
 public:
  struct CtorArgs {
    VariantCode variant_kind{};
    bool can_raise_exception{};
    bool can_elide_device_ctx_sync{};
    std::vector<InternalSharedPtr<PhysicalArray>> inputs{};
    std::vector<InternalSharedPtr<PhysicalArray>> outputs{};
    std::vector<InternalSharedPtr<PhysicalArray>> reductions{};
    std::vector<InternalSharedPtr<Scalar>> scalars{};
    std::vector<legate::comm::Communicator> comms{};
  };

  explicit TaskContext(CtorArgs&& args);

  [[nodiscard]] const std::vector<InternalSharedPtr<PhysicalArray>>& inputs() const noexcept;
  [[nodiscard]] const std::vector<InternalSharedPtr<PhysicalArray>>& outputs() const noexcept;
  [[nodiscard]] const std::vector<InternalSharedPtr<PhysicalArray>>& reductions() const noexcept;
  [[nodiscard]] const std::vector<InternalSharedPtr<Scalar>>& scalars() const noexcept;
  [[nodiscard]] const std::vector<legate::comm::Communicator>& communicators() const noexcept;

  [[nodiscard]] virtual GlobalTaskID task_id() const noexcept                    = 0;
  [[nodiscard]] virtual bool is_single_task() const noexcept                     = 0;
  [[nodiscard]] virtual const DomainPoint& get_task_index() const noexcept       = 0;
  [[nodiscard]] virtual const Domain& get_launch_domain() const noexcept         = 0;
  [[nodiscard]] virtual std::string_view get_provenance() const                  = 0;
  [[nodiscard]] virtual const mapping::detail::Machine& machine() const noexcept = 0;

  [[nodiscard]] VariantCode variant_kind() const noexcept;
  [[nodiscard]] bool can_raise_exception() const noexcept;
  [[nodiscard]] CUstream get_task_stream() const;
  [[nodiscard]] bool can_elide_device_ctx_sync() const noexcept;

  void set_exception(ReturnedException what);
  [[nodiscard]] std::optional<ReturnedException>& get_exception() noexcept;

  /**
   * @brief Makes all of unbound output stores of this task empty
   */
  void make_all_unbound_stores_empty();

  void concurrent_task_barrier();

 protected:
  [[nodiscard]] const std::vector<InternalSharedPtr<PhysicalStore>>& get_unbound_stores_()
    const noexcept;
  [[nodiscard]] const std::vector<InternalSharedPtr<PhysicalStore>>& get_scalar_stores_()
    const noexcept;

 private:
  VariantCode variant_kind_{};
  std::vector<InternalSharedPtr<PhysicalArray>> inputs_{};
  std::vector<InternalSharedPtr<PhysicalArray>> outputs_{};
  std::vector<InternalSharedPtr<PhysicalArray>> reductions_{};
  std::vector<InternalSharedPtr<PhysicalStore>> unbound_stores_{};
  std::vector<InternalSharedPtr<PhysicalStore>> scalar_stores_{};
  std::vector<InternalSharedPtr<Scalar>> scalars_{};
  std::vector<legate::comm::Communicator> comms_{};
  bool can_raise_exception_{};
  bool can_elide_device_ctx_sync_{};
  std::optional<ReturnedException> excn_{std::nullopt};
};

}  // namespace legate::detail

#include <legate/task/detail/task_context.inl>
