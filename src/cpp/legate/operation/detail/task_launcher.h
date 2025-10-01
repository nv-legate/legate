/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/scalar.h>
#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/runtime/detail/streaming.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/zstring_view.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace legate::detail {

class Library;
class Runtime;

class TaskLauncher {
 public:
  TaskLauncher(const Library& library,
               const mapping::detail::Machine& machine,
               const ParallelPolicy& parallel_policy,
               ZStringView provenance,
               LocalTaskID task_id,
               Legion::MappingTagID tag = 0);

  TaskLauncher(const Library& library,
               const mapping::detail::Machine& machine,
               const ParallelPolicy& parallel_policy,
               LocalTaskID task_id,
               Legion::MappingTagID tag = 0);

  [[nodiscard]] GlobalTaskID legion_task_id() const;

  /**
   * @brief Reserves space for a specified number of input elements.
   *
   * @param num The number of input elements to reserve space for.
   */
  void reserve_inputs(std::size_t num);
  void add_input(Analyzable arg);

  /**
   * @brief Reserves space for a specified number of output elements.
   *
   * @param num The number of output elements to reserve space for.
   */
  void reserve_outputs(std::size_t num);
  void add_output(Analyzable arg);

  /**
   * @brief Reserves space for a specified number of reduction elements.
   *
   * @param num The number of reduction elements to reserve space for.
   */
  void reserve_reductions(std::size_t num);
  void add_reduction(Analyzable arg);

  /**
   * @brief Reserves space for a specified number of scalar elements.
   *
   * @param num The number of scalar elements to reserve space for.
   */
  void reserve_scalars(std::size_t num);
  void add_scalar(InternalSharedPtr<Scalar> scalar);

  void add_future(Legion::Future future);
  void add_future_map(Legion::FutureMap future_map);

  /**
   * @brief Reserves space for a specified number of communicators..
   *
   * @param num The number of communicators to reserve space for.
   */
  void reserve_communicators(std::size_t num);
  void add_communicator(Legion::FutureMap communicator);

  void set_priority(std::int32_t priority);
  void set_side_effect(bool has_side_effect);
  void set_concurrent(bool is_concurrent);
  void set_insert_barrier(bool insert_barrier);
  /**
   * @brief Set the maximum future size of this task.
   */
  void set_future_size(std::size_t future_size);
  void throws_exception(bool can_throw_exception);
  void can_elide_device_ctx_sync(bool can_elide_sync);
  void relax_interference_checks(bool relax);

  /**
   * @brief Set the streaming generation for this particular task launch.
   *
   * See `Task::set_streaming_generation()` for further discussion on this parameter.
   *
   * @param streaming_generation The streaming generation.
   */
  void set_streaming_generation(const std::optional<StreamingGeneration>& streaming_generation);

  Legion::FutureMap execute(const Legion::Domain& launch_domain);
  Legion::Future execute_single();

  [[nodiscard]] ZStringView provenance() const;
  [[nodiscard]] const ParallelPolicy& parallel_policy() const;

  /**
   * @return The streaming generation if this task is a streaming task, `std::nullopt` otherwise.
   */
  [[nodiscard]] const std::optional<StreamingGeneration>& streaming_generation() const;

 private:
  void analyze_arguments_(bool parallel, StoreAnalyzer* analyzer);
  BufferBuilder pack_task_arg_(bool parallel, StoreAnalyzer* analyzer);
  void pack_mapper_arg_(BufferBuilder& buffer);
  void import_output_regions_(Runtime& runtime,
                              const std::vector<Legion::OutputRequirement>& output_requirements);
  void post_process_unbound_stores_(
    const std::vector<Legion::OutputRequirement>& output_requirements);
  void post_process_unbound_stores_(
    const Legion::FutureMap& result,
    const Legion::Domain& launch_domain,
    const std::vector<Legion::OutputRequirement>& output_requirements);

  void post_process_unbound_store_(const Legion::Domain& launch_domain,
                                   const OutputRegionArg* arg,
                                   const Legion::OutputRequirement& req,
                                   const Legion::FutureMap& weights,
                                   const mapping::detail::Machine& machine,
                                   const ParallelPolicy& parallel_policy);

  void report_interfering_stores_() const;
  /**
   * @brief Get the maximum future size set by the caller.
   */
  [[nodiscard]] std::size_t get_future_size_() const;

  std::reference_wrapper<const Library> library_;
  LocalTaskID task_id_{};
  Legion::MappingTagID tag_{};
  const mapping::detail::Machine& machine_;
  ParallelPolicy parallel_policy_{};
  ZStringView provenance_{};
  std::int32_t priority_{static_cast<std::int32_t>(TaskPriority::DEFAULT)};

  bool has_side_effect_{true};
  bool concurrent_{};
  bool insert_barrier_{};
  bool can_throw_exception_{};
  bool can_elide_device_ctx_sync_{};
  bool relax_interference_checks_{};
  std::optional<StreamingGeneration> streaming_gen_{};
  std::size_t future_size_{};

  std::vector<Analyzable> inputs_{};
  std::vector<Analyzable> outputs_{};
  std::vector<Analyzable> reductions_{};
  std::vector<ScalarArg> scalars_{};
  std::vector<Legion::Future> futures_{};
  std::vector<Legion::FutureMap> future_maps_{};
  std::vector<Legion::FutureMap> communicators_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/task_launcher.inl>
