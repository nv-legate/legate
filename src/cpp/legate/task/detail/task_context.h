/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/communicator.h>
#include <legate/cuda/detail/cuda_driver_types.h>
#include <legate/data/scalar.h>
#include <legate/mapping/detail/machine.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <optional>
#include <string_view>

namespace legate::detail {

class PhysicalStore;
class TaskBase;

class TaskContext {
 public:
  TaskContext(VariantCode variant_code,
              const Domain& launch_domain,
              const DomainPoint& task_index,
              const TaskBase* task,
              SmallVector<InternalSharedPtr<PhysicalStore>> inputs,
              SmallVector<InternalSharedPtr<PhysicalStore>> outputs,
              SmallVector<InternalSharedPtr<PhysicalStore>> reductions,
              SmallVector<InternalSharedPtr<Scalar>> scalars);

  [[nodiscard]] Span<const InternalSharedPtr<PhysicalStore>> inputs() const noexcept;
  [[nodiscard]] Span<const InternalSharedPtr<PhysicalStore>> outputs() const noexcept;
  [[nodiscard]] Span<const InternalSharedPtr<PhysicalStore>> reductions() const noexcept;
  [[nodiscard]] Span<const InternalSharedPtr<Scalar>> scalars() const noexcept;
  [[nodiscard]] Span<const legate::comm::Communicator> communicators() const noexcept;

  [[nodiscard]] LocalTaskID task_id() const noexcept;
  [[nodiscard]] std::string_view library_name() const noexcept;
  [[nodiscard]] bool is_single_task() const noexcept;
  [[nodiscard]] const DomainPoint& get_task_index() const noexcept;
  [[nodiscard]] const Domain& get_launch_domain() const noexcept;
  [[nodiscard]] std::string_view get_provenance() const noexcept;
  [[nodiscard]] const mapping::detail::Machine& machine() const noexcept;

  [[nodiscard]] VariantCode variant_kind() const noexcept;
  [[nodiscard]] bool can_raise_exception() const noexcept;
  [[nodiscard]] CUstream get_task_stream() const;
  [[nodiscard]] bool can_elide_device_ctx_sync() const noexcept;

  /**
   * @brief Makes all of unbound output stores of this task empty
   */
  void make_all_unbound_stores_empty();

  void concurrent_task_barrier();

 protected:
  [[nodiscard]] Span<const InternalSharedPtr<PhysicalStore>> get_unbound_stores_() const noexcept;
  [[nodiscard]] Span<const InternalSharedPtr<PhysicalStore>> get_scalar_stores_() const noexcept;

 private:
  const VariantCode variant_kind_{};
  const Domain launch_domain_{};
  const DomainPoint task_index_{};
  const TaskBase* op_task_{};
  SmallVector<InternalSharedPtr<PhysicalStore>> inputs_{};
  SmallVector<InternalSharedPtr<PhysicalStore>> outputs_{};
  SmallVector<InternalSharedPtr<PhysicalStore>> reductions_{};
  SmallVector<InternalSharedPtr<PhysicalStore>> unbound_stores_{};
  SmallVector<InternalSharedPtr<PhysicalStore>> scalar_stores_{};
  SmallVector<InternalSharedPtr<Scalar>> scalars_{};
  SmallVector<legate::comm::Communicator> comms_{};
};

}  // namespace legate::detail

#include <legate/task/detail/task_context.inl>
