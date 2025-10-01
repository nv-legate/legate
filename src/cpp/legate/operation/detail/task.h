/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_array.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/scalar.h>
#include <legate/operation/detail/operation.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/streaming.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace legate {

class Scalar;

}  // namespace legate

namespace legate::detail {

class CommunicatorFactory;
class ConstraintSolver;
class Library;
class VariantInfo;

class TaskArrayArg {
 public:
  /**
   * @brief Construct a TaskArrayArg.
   *
   * `priv` should be initialized to the following values based on whether the argument as an
   * input, output, or reduction:
   *
   * - input: `LEGION_READ_ONLY`
   * - output: `LEGION_WRITE_ONLY`
   * - reduction: `LEGION_REDUCE`
   *
   * If the owning task is a streaming task, then this privilege is further fixed up during
   * scheduling window flush to include additional discard privileges. Therefore, the privilege
   * member should *not* be considered stable until the task is sent to Legion.
   *
   * @param priv The access privilege for this task argument.
   * @param _array The array for this argument.
   * @param _projection An optional projection for the argument.
   */
  TaskArrayArg(Legion::PrivilegeMode priv,
               InternalSharedPtr<LogicalArray> _array,
               std::optional<SymbolicPoint> _projection = std::nullopt);
  [[nodiscard]] bool needs_flush() const;

  Legion::PrivilegeMode privilege{Legion::PrivilegeMode::LEGION_NO_ACCESS};
  InternalSharedPtr<LogicalArray> array{};
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*> mapping{};
  std::optional<SymbolicPoint> projection{};
};

class Task : public Operation {
 protected:
  Task(const Library& library,
       const VariantInfo& variant_info,
       LocalTaskID task_id,
       std::uint64_t unique_id,
       std::int32_t priority,
       mapping::detail::Machine machine,
       bool can_inline_launch);

 public:
  void add_scalar_arg(InternalSharedPtr<Scalar> scalar);
  void set_concurrent(bool concurrent);
  void set_side_effect(bool has_side_effect);
  void throws_exception(bool can_throw_exception);
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

  void record_scalar_output(InternalSharedPtr<LogicalStore> store);
  void record_unbound_output(InternalSharedPtr<LogicalStore> store);
  void record_scalar_reduction(InternalSharedPtr<LogicalStore> store,
                               GlobalRedopID legion_redop_id);

  void validate() override;

 protected:
  void launch_task_(Strategy* strategy);

 private:
  void inline_launch_() const;
  void legion_launch_(Strategy* strategy);

  /**
   * @brief De-multiplex a future returned by a Legion task.
   *
   * Because Legion allows each task to have only up to one returned future, Legate packs multiple
   * scalars it needs to return from a Legate task into the single future, which later gets
   * de-multiplexed by this method. Scalar output and reduction stores directly use the returned
   * future as their backing storage, and whoever consume them offset into the right location in
   * that future. The returned future contains a serialized exception at the end, which needs to be
   * extracted out and converted into an exception object of the right type so it can be re-raised
   * on the control side.
   */
  void demux_scalar_stores_(const Legion::Future& result, std::size_t future_size_without_exn);
  /**
   * @brief De-multiplex a future map returned by a Legion task.
   *
   * This method is an "index version" of the `demux_scalar_stores` method. This does more or less
   * the same thing, except that it needs to reduce multiple scalars into a single scalar whenever
   * necessary; for scalar reduction stores, the method extracts scalars holding local reduction
   * contributions from parallel tasks (using the same `ExtractScalar` task) and passes the future
   * map to Legion for the scalar reduction producing a final output; returned exceptions also need
   * to be combined in a similar way, and the "reduction" operator for returned exceptions simply
   * favors the previous one over all the later ones (i.e., if a task i returned an exception, those
   * returned from all tasks j > i are ignored.
   */
  void demux_scalar_stores_(const Legion::FutureMap& result,
                            const Domain& launch_domain,
                            std::size_t future_size_without_exn);

  // Calculate the return future size excluding the size of returned exception, which can only be
  // approximate
  [[nodiscard]] std::size_t calculate_future_size_() const;

 public:
  [[nodiscard]] std::string to_string(bool show_provenance) const override;
  [[nodiscard]] bool needs_flush() const override;
  [[nodiscard]] bool supports_replicated_write() const override;
  [[nodiscard]] bool can_throw_exception() const;
  [[nodiscard]] bool can_elide_device_ctx_sync() const;

  /**
   * @return The streaming generation if the task is a streaming task, `std::nullopt` otherwise.
   */
  [[nodiscard]] const std::optional<StreamingGeneration>& streaming_generation() const;

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

  /**
   * @return The scalar output arguments for the task.
   */
  [[nodiscard]] Span<const InternalSharedPtr<LogicalStore>> scalar_outputs() const;

  /**
   * @return The scalar reductions for the task.
   */
  [[nodiscard]] Span<const std::pair<InternalSharedPtr<LogicalStore>, GlobalRedopID>>
  scalar_reductions() const;

  [[nodiscard]] const Library& library() const;
  [[nodiscard]] LocalTaskID local_task_id() const;

 protected:
  [[nodiscard]] const VariantInfo& variant_info_() const;

 private:
  std::reference_wrapper<const Library> library_;
  std::reference_wrapper<const VariantInfo> vinfo_;
  LocalTaskID task_id_{};
  bool concurrent_{};
  bool has_side_effect_{};
  bool can_throw_exception_{};
  bool can_elide_device_ctx_sync_{};
  std::optional<StreamingGeneration> streaming_gen_{};
  SmallVector<InternalSharedPtr<Scalar>> scalars_{};

 protected:
  SmallVector<TaskArrayArg> inputs_{};
  SmallVector<TaskArrayArg> outputs_{};
  SmallVector<TaskArrayArg> reductions_{};
  SmallVector<GlobalRedopID> reduction_ops_{};

 private:
  SmallVector<InternalSharedPtr<LogicalStore>> unbound_outputs_{};
  SmallVector<InternalSharedPtr<LogicalStore>> scalar_outputs_{};
  SmallVector<std::pair<InternalSharedPtr<LogicalStore>, GlobalRedopID>> scalar_reductions_{};
  SmallVector<std::reference_wrapper<CommunicatorFactory>> communicator_factories_{};
  bool can_inline_launch_{};
};

class AutoTask final : public Task {
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

class ManualTask final : public Task {
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

 private:
  Strategy strategy_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/task.inl>
