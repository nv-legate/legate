/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_array.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/data/detail/scalar.h>
#include <legate/operation/detail/operation.h>
#include <legate/operation/detail/task_array_arg.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/streaming.h>
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
           mapping::detail::Machine machine,
           bool can_inline_launch);

 public:
  void validate() override;
  void add_scalar_arg(InternalSharedPtr<Scalar> scalar);
  void set_concurrent(bool concurrent);
  void set_side_effect(bool has_side_effect);
  void throws_exception(bool can_throw_exception);

 protected:
  void inline_launch_() const;

 public:
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
  [[nodiscard]] Span<const TaskArrayArg> inputs() const;

  /**
   * @return The output arguments for the task.
   */
  [[nodiscard]] Span<const TaskArrayArg> outputs() const;

  /**
   * @return The reduction arguments for the task.
   */
  [[nodiscard]] Span<const TaskArrayArg> reductions() const;

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
  SmallVector<TaskArrayArg> inputs_{};
  SmallVector<TaskArrayArg> outputs_{};
  SmallVector<TaskArrayArg> reductions_{};
  SmallVector<GlobalRedopID> reduction_ops_{};
  bool can_inline_launch_{};
};

class LogicalTask : public TaskBase {
 protected:
  LogicalTask(const Library& library,
              const VariantInfo& variant_info,
              LocalTaskID task_id,
              std::uint64_t unique_id,
              std::int32_t priority,
              mapping::detail::Machine machine,
              bool can_inline_launch);

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
  void add_communicator(std::string_view name, bool bypass_signature_check = false);

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
  void legion_launch_(Strategy* strategy);
  void demux_scalar_stores_(const Legion::Future& result, std::size_t future_size);
  void demux_scalar_stores_(const Legion::FutureMap& result,
                            const Domain& launch_domain,
                            std::size_t future_size);
  [[nodiscard]] std::size_t calculate_future_size_() const;

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

  [[nodiscard]] const Variable* add_input(InternalSharedPtr<LogicalArray> array);
  [[nodiscard]] const Variable* add_output(InternalSharedPtr<LogicalArray> array);
  [[nodiscard]] const Variable* add_reduction(InternalSharedPtr<LogicalArray> array,
                                              std::int32_t redop_kind);

  void add_input(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol);
  void add_output(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol);
  void add_reduction(InternalSharedPtr<LogicalArray> array,
                     std::int32_t redop_kind,
                     const Variable* partition_symbol);

  [[nodiscard]] const Variable* find_or_declare_partition(
    const InternalSharedPtr<LogicalArray>& array);

  void add_constraint(InternalSharedPtr<Constraint> constraint,
                      bool bypass_signature_check = false);
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
  void fixup_ranges_(Strategy& strategy);

  SmallVector<InternalSharedPtr<Constraint>> constraints_{};
  SmallVector<LogicalArray*> arrays_to_fixup_{};
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

  void add_input(const InternalSharedPtr<LogicalStore>& store);
  void add_input(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                 std::optional<SymbolicPoint> projection);
  void add_output(const InternalSharedPtr<LogicalStore>& store);
  void add_output(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                  std::optional<SymbolicPoint> projection);
  void add_reduction(const InternalSharedPtr<LogicalStore>& store, std::int32_t redop_kind);
  void add_reduction(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                     std::int32_t redop_kind,
                     std::optional<SymbolicPoint> projection);

  /**
   * @brief Get the launch domain for this ManualTask.
   *
   * @return The launch domain for the task.
   */
  [[nodiscard]] const Domain& launch_domain() const;

 private:
  /**
   * @brief Add a store as an argument to the `ManualTask`
   *
   * @param priv The privilege of the store to add. See `TaskArrayArg` for further discussion
   * on this value.
   * @param store_args The array of arguments to append to.
   * @param store The store to add as an argument.
   * @param partition The partitioning descriptor for the store.
   * @param projection An optional projection to use on the partitioned store.
   */
  void add_store_(Legion::PrivilegeMode priv,
                  SmallVector<TaskArrayArg>& store_args,
                  const InternalSharedPtr<LogicalStore>& store,
                  InternalSharedPtr<Partition> partition,
                  std::optional<SymbolicPoint> projection = {});

 public:
  void launch() override;

  [[nodiscard]] Kind kind() const override;

  /**
   * @return `false`, `ManualTask` operations by definition have the runtime compute the
   * partitioning for each argument.
   */
  [[nodiscard]] bool needs_partitioning() const override;

  /**
   * @see Operation::supports_streaming
   */
  [[nodiscard]] bool supports_streaming() const override;

  /**
   * Provide a copy of internal Strategy
   */
  [[nodiscard]] Strategy copy_strategy() const;

 private:
  Strategy strategy_{};
};

/**
 * @brief A task that operates directly on physical data arrays with explicit memory layouts.
 *
 * PhysicalTask represents the leaf level of task execution in Legate, where tasks work
 * directly with physical memory representations of data. Unlike AutoTask and ManualTask
 * which work with logical arrays, PhysicalTask requires explicit physical array inputs
 * and outputs with known memory layouts and partitioning.
 *
 * Physical tasks are typically created by top-level tasks (AutoTask/ManualTask) during their
 * execution phase when the logical-to-physical mapping has already been determined by the
 * runtime.
 */
class PhysicalTask final : public TaskBase {
 public:
  /**
   * @brief Constructs a new PhysicalTask.
   *
   * @param library The library containing the task implementation
   * @param variant_info Information about the specific task variant to execute
   * @param task_id Local task identifier within the library
   * @param unique_id Globally unique identifier for this task instance
   * @param machine Target machine/processor for task execution
   */
  PhysicalTask(const Library& library,
               const VariantInfo& variant_info,
               LocalTaskID task_id,
               std::uint64_t unique_id,
               mapping::detail::Machine machine);

  /**
   * @brief Adds a read-only input array to the task.
   *
   * The input array will be available for reading during task execution.
   * The array's physical layout and partitioning must already be determined.
   *
   * @param array Physical array to add as input with read-only access
   */
  void add_input(InternalSharedPtr<PhysicalArray> array);

  /**
   * @brief Adds a write-only output array to the task.
   *
   * The output array will be available for writing during task execution.
   * The array's physical layout and partitioning must already be determined.
   *
   * @param array Physical array to add as output with write-only access
   */
  void add_output(InternalSharedPtr<PhysicalArray> array);

  /**
   * @brief Adds a reduction array to the task with a specific reduction operator.
   *
   * The reduction array will be available for reduction operations during task execution.
   * Multiple tasks can perform reductions on the same array, with the runtime handling
   * the combination of partial results using the specified reduction operator.
   *
   * @param array Physical array to add as reduction target
   * @param redop_kind Reduction operator kind (e.g., sum, max, min, product)
   */
  void add_reduction(InternalSharedPtr<PhysicalArray> array, std::int32_t redop_kind);

  // Note: PhysicalTask uses physical_scalar_outputs() and physical_scalar_reductions()
  // instead of the LogicalStore-based methods in LogicalTask

  /**
   * @brief Launches the physical task for execution.
   *
   * Submits the task to the Legion runtime for execution on the target machine.
   * All input, output, and reduction arrays must be added before launching.
   * This is a non-blocking operation that returns immediately.
   *
   * @throws std::runtime_error if the task cannot be launched
   */
  void launch() override;

  /**
   * @brief Returns the kind of task.
   *
   * @return Always returns Kind::PHYSICAL for PhysicalTask instances
   */
  [[nodiscard]] Kind kind() const override;

  /**
   * @brief Indicates whether this task needs runtime partitioning.
   *
   * PhysicalTask operations work with pre-partitioned physical arrays, so they
   * do not require additional partitioning by the runtime.
   *
   * @return Always returns `false` for PhysicalTask instances
   */
  [[nodiscard]] bool needs_partitioning() const override;

  /**
   * @brief Adds a scalar output to the task.
   *
   * Scalar outputs are backed by futures and need special handling during task completion.
   * The scalar will be available as a future after task execution completes.
   *
   * @param store Physical store representing the scalar output
   */
  void add_scalar_output(InternalSharedPtr<PhysicalStore> store);

  /**
   * @brief Adds a scalar reduction to the task.
   *
   * Scalar reductions combine results from multiple task instances using the specified
   * reduction operator. The final result will be available as a future.
   *
   * @param store Physical store representing the scalar reduction
   * @param redop_id Global reduction operator ID
   */
  void add_scalar_reduction(InternalSharedPtr<PhysicalStore> store, GlobalRedopID redop_id);

  /**
   * @brief Access PhysicalTask's scalar outputs (PhysicalStore-based).
   *
   * This method provides access to the PhysicalStore-based scalar outputs,
   * which is needed by the template specialization in handle_return_values.
   *
   * @return Span of PhysicalStore scalar outputs
   */
  [[nodiscard]] Span<const InternalSharedPtr<PhysicalStore>> physical_scalar_outputs() const;

  /**
   * @brief Access PhysicalTask's scalar reductions (PhysicalStore-based).
   *
   * This method provides access to the PhysicalStore-based scalar reductions,
   * which is needed by the template specialization in handle_return_values.
   *
   * @return Span of PhysicalStore scalar reductions with their reduction operators
   */
  [[nodiscard]] Span<const std::pair<InternalSharedPtr<PhysicalStore>, GlobalRedopID>>
  physical_scalar_reductions() const;

 private:
  void fixup_ranges_(Strategy& strategy);

  // Storage for scalar outputs and reductions (PhysicalTask-specific)
  SmallVector<InternalSharedPtr<PhysicalStore>> scalar_outputs_{};
  SmallVector<std::pair<InternalSharedPtr<PhysicalStore>, GlobalRedopID>> scalar_reductions_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/task.inl>
