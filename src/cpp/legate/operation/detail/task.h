/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/data/detail/scalar.h>
#include <legate/mapping/mapping.h>
#include <legate/operation/detail/operation.h>
#include <legate/operation/detail/task_store_arg.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/streaming/generation.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace legate {

class Scalar;

}  // namespace legate

namespace legate::detail {

class PhysicalStore;
class CommunicatorFactory;
class ConstraintSolver;
class Library;
class VariantInfo;

class TaskBase : public Operation {
 protected:
  TaskBase(const Library& library,
           const VariantInfo& variant_info,
           LocalTaskID task_id,
           std::uint64_t unique_id,
           std::int32_t priority,
           mapping::detail::Machine machine);

 public:
  void validate() override;
  void add_scalar_arg(InternalSharedPtr<Scalar> scalar);
  void set_concurrent(bool concurrent);
  void set_side_effect(bool has_side_effect);
  void throws_exception(bool can_throw_exception);

  [[nodiscard]] std::string to_string(bool show_provenance) const override;
  [[nodiscard]] bool needs_flush() const override;
  [[nodiscard]] bool supports_replicated_write() const override;
  [[nodiscard]] bool can_throw_exception() const;
  [[nodiscard]] bool can_elide_device_ctx_sync() const;

  /**
   * @return The scalar arguments for the task.
   */
  [[nodiscard]] Span<const InternalSharedPtr<Scalar>> scalars() const;

  /**
   * @return The input arguments for the task.
   */
  [[nodiscard]] Span<const TaskStoreArg> inputs() const;

  /**
   * @return The output arguments for the task.
   */
  [[nodiscard]] Span<const TaskStoreArg> outputs() const;

  /**
   * @return The reduction arguments for the task.
   */
  [[nodiscard]] Span<const TaskStoreArg> reductions() const;

  [[nodiscard]] const Library& library() const;
  [[nodiscard]] LocalTaskID local_task_id() const;

 protected:
  [[nodiscard]] const VariantInfo& variant_info_() const;

  std::reference_wrapper<const Library> library_;
  std::reference_wrapper<const VariantInfo> vinfo_;
  LocalTaskID task_id_{};
  bool concurrent_{};
  bool has_side_effect_{};
  bool can_throw_exception_{};
  bool can_elide_device_ctx_sync_{};
  SmallVector<InternalSharedPtr<Scalar>> scalars_{};
  SmallVector<TaskStoreArg> inputs_{};
  SmallVector<TaskStoreArg> outputs_{};
  SmallVector<TaskStoreArg> reductions_{};
  SmallVector<GlobalRedopID> reduction_ops_{};
};

class LogicalTask : public TaskBase {
 protected:
  LogicalTask(const Library& library,
              const VariantInfo& variant_info,
              LocalTaskID task_id,
              std::uint64_t unique_id,
              std::int32_t priority,
              mapping::detail::Machine machine);

 public:
  // LogicalTask-specific scalar methods (non-virtual)
  [[nodiscard]] Span<const InternalSharedPtr<LogicalStore>> scalar_outputs() const;
  [[nodiscard]] Span<const std::pair<InternalSharedPtr<LogicalStore>, GlobalRedopID>>
  scalar_reductions() const;

  /**
   * @brief Set the streaming generation for this task.
   *
   * This marks a task as part of a set of streaming tasks. Among other effects, this informs
   * the mapper that this task should be mapped "vertically", i.e. map each "column" of index
   * points of a streaming generation at a time before mapping new tasks. See
   * `BaseMapper::select_tasks_to_map()` for more information.
   *
   * If `streaming_gen` is `std::nullopt`, the task is no longer considered to be part of a
   * streaming generation. It will be eagerly (i.e. horizontally) mapped by the mapper.
   *
   * @param streaming_gen The streaming generation.
   */
  void set_streaming_generation(std::optional<StreamingGeneration> streaming_gen);
  void add_communicator(std::string_view name, bool bypass_signature_check);

  /**
   * @return The streaming generation if the task is a streaming task, `std::nullopt` otherwise.
   */
  [[nodiscard]] const std::optional<StreamingGeneration>& streaming_generation() const;

  void record_scalar_output(InternalSharedPtr<LogicalStore> store);
  void record_unbound_output(InternalSharedPtr<LogicalStore> store);
  void record_scalar_reduction(InternalSharedPtr<LogicalStore> store,
                               GlobalRedopID legion_redop_id);

 protected:
  void launch_task_(Strategy* strategy);

 private:
  SmallVector<InternalSharedPtr<LogicalStore>> unbound_outputs_{};
  SmallVector<InternalSharedPtr<LogicalStore>> scalar_outputs_{};
  SmallVector<std::pair<InternalSharedPtr<LogicalStore>, GlobalRedopID>> scalar_reductions_{};
  SmallVector<std::reference_wrapper<CommunicatorFactory>> communicator_factories_{};
  std::optional<StreamingGeneration> streaming_gen_{};
};

class AutoTask final : public LogicalTask {
 public:
  AutoTask(const Library& library,
           const VariantInfo& variant_info,
           LocalTaskID task_id,
           std::uint64_t unique_id,
           std::int32_t priority,
           mapping::detail::Machine machine);

  [[nodiscard]] const Variable* add_input(InternalSharedPtr<LogicalStore> store);
  [[nodiscard]] const Variable* add_output(InternalSharedPtr<LogicalStore> store);
  [[nodiscard]] const Variable* add_reduction(InternalSharedPtr<LogicalStore> store,
                                              std::int32_t redop_kind);

  void add_input(InternalSharedPtr<LogicalStore> store, const Variable* partition_symbol);
  void add_output(InternalSharedPtr<LogicalStore> store, const Variable* partition_symbol);
  void add_reduction(InternalSharedPtr<LogicalStore> store,
                     std::int32_t redop_kind,
                     const Variable* partition_symbol);

  void add_constraint(InternalSharedPtr<Constraint> constraint, bool bypass_signature_check);
  void add_to_solver(ConstraintSolver& solver) override;

  void validate() override;
  void launch(Strategy* strategy) override;

  [[nodiscard]] Kind kind() const override;

  /**
   * @return `true`, `AutoTask` operations by definition have the runtime compute the
   * partitioning for each argument.
   */
  [[nodiscard]] bool needs_partitioning() const override;

 private:
  SmallVector<InternalSharedPtr<Constraint>> constraints_{};
};

class ManualTask final : public LogicalTask {
 public:
  ManualTask(const Library& library,
             const VariantInfo& variant_info,
             LocalTaskID task_id,
             const Domain& launch_domain,
             std::uint64_t unique_id,
             std::int32_t priority,
             mapping::detail::Machine machine);

  void add_input(InternalSharedPtr<LogicalStore> store);
  void add_output(InternalSharedPtr<LogicalStore> store);
  void add_reduction(InternalSharedPtr<LogicalStore> store, std::int32_t redop_kind);
  void add_input(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                 std::optional<SymbolicPoint> projection,
                 bool is_key_partition);
  void add_output(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                  std::optional<SymbolicPoint> projection,
                  bool is_key_partition);
  void add_reduction(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                     std::int32_t redop_kind,
                     std::optional<SymbolicPoint> projection,
                     bool is_key_partition);

  /**
   * @brief Get the launch domain for this ManualTask.
   *
   * @return The launch domain for the task.
   */
  [[nodiscard]] const Domain& launch_domain() const;

  void launch() override;

  [[nodiscard]] Kind kind() const override;

  /**
   * @return `false`, `ManualTask` operations by definition have the runtime compute the
   * partitioning for each argument.
   */
  [[nodiscard]] bool needs_partitioning() const override;

  /**
   * @brief @see Operation::supports_streaming.
   *
   * @return true.
   */
  [[nodiscard]] bool supports_streaming() const override;

  /**
   * @brief used by streaming to read strategy of a manual task.
   * Assumes strategy has been computed.
   *
   * @return strategy by shared pointer.
   */
  [[nodiscard]] const InternalSharedPtr<Strategy>& strategy() const;

 private:
  /**
   * @brief Add a store as an argument to the `ManualTask`
   *
   * @param priv The privilege of the store to add. See `TaskStoreArg` for further discussion
   * on this value.
   * @param store_args The store of arguments to append to.
   * @param store The store to add as an argument.
   * @param partition The partitioning descriptor for the store.
   * @param projection An optional projection to use on the partitioned store.
   */
  void add_store_(Legion::PrivilegeMode priv,
                  SmallVector<TaskStoreArg>& store_args,
                  const InternalSharedPtr<LogicalStore>& store,
                  InternalSharedPtr<Partition> partition,
                  std::optional<SymbolicPoint> projection = {},
                  bool is_key_partition                   = false);

  InternalSharedPtr<Strategy> strategy_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/task.inl>
