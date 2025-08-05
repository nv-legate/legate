/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/instance_manager.h>
#include <legate/mapping/detail/machine.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/typedefs.h>

#include <legion.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace legate::detail {

class StreamingGeneration;

}  // namespace legate::detail

namespace legate::mapping::detail {

class BaseMapper final : public Legion::Mapping::Mapper, public MachineQueryInterface {
 public:
  static constexpr std::string_view LOGGER_NAME = "legate.mapper";

  BaseMapper();

  ~BaseMapper() override;

  BaseMapper(const BaseMapper& rhs)            = delete;
  BaseMapper& operator=(const BaseMapper& rhs) = delete;

  [[nodiscard]] Legion::Logger& logger();
  [[nodiscard]] const Legion::Logger& logger() const;

  // MachineQueryInterface
  [[nodiscard]] const std::vector<Processor>& cpus() const override;
  [[nodiscard]] const std::vector<Processor>& gpus() const override;
  [[nodiscard]] const std::vector<Processor>& omps() const override;
  [[nodiscard]] std::uint32_t total_nodes() const override;
  [[nodiscard]] const char* get_mapper_name() const override;
  [[nodiscard]] Legion::Mapping::Mapper::MapperSyncModel get_mapper_sync_model() const override;

  [[nodiscard]] bool request_valid_instances() const override;

  // Task mapping calls
  void select_task_options(Legion::Mapping::MapperContext ctx,
                           const Legion::Task& task,
                           TaskOptions& output) override;
  void premap_task(Legion::Mapping::MapperContext ctx,
                   const Legion::Task& task,
                   const PremapTaskInput& input,
                   PremapTaskOutput& output) override;
  void slice_task(Legion::Mapping::MapperContext ctx,
                  const Legion::Task& task,
                  const SliceTaskInput& input,
                  SliceTaskOutput& output) override;
  void map_task(Legion::Mapping::MapperContext ctx,
                const Legion::Task& task,
                const MapTaskInput& input,
                MapTaskOutput& output) override;
  void replicate_task(Legion::Mapping::MapperContext ctx,
                      const Legion::Task& task,
                      const ReplicateTaskInput& input,
                      ReplicateTaskOutput& output) override;
  void select_task_variant(Legion::Mapping::MapperContext ctx,
                           const Legion::Task& task,
                           const SelectVariantInput& input,
                           SelectVariantOutput& output) override;
  void postmap_task(Legion::Mapping::MapperContext ctx,
                    const Legion::Task& task,
                    const PostMapInput& input,
                    PostMapOutput& output) override;
  void select_task_sources(Legion::Mapping::MapperContext ctx,
                           const Legion::Task& task,
                           const SelectTaskSrcInput& input,
                           SelectTaskSrcOutput& output) override;
  void report_profiling(Legion::Mapping::MapperContext ctx,
                        const Legion::Task& task,
                        const TaskProfilingInfo& input) override;
  void select_sharding_functor(Legion::Mapping::MapperContext ctx,
                               const Legion::Task& task,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override;

  // Inline mapping calls
  void map_inline(Legion::Mapping::MapperContext ctx,
                  const Legion::InlineMapping& inline_op,
                  const MapInlineInput& input,
                  MapInlineOutput& output) override;
  void select_inline_sources(Legion::Mapping::MapperContext ctx,
                             const Legion::InlineMapping& inline_op,
                             const SelectInlineSrcInput& input,
                             SelectInlineSrcOutput& output) override;
  void report_profiling(Legion::Mapping::MapperContext ctx,
                        const Legion::InlineMapping& inline_op,
                        const InlineProfilingInfo& input) override;

  // Copy mapping calls
  void map_copy(Legion::Mapping::MapperContext ctx,
                const Legion::Copy& copy,
                const MapCopyInput& input,
                MapCopyOutput& output) override;
  void select_copy_sources(Legion::Mapping::MapperContext ctx,
                           const Legion::Copy& copy,
                           const SelectCopySrcInput& input,
                           SelectCopySrcOutput& output) override;
  void report_profiling(Legion::Mapping::MapperContext ctx,
                        const Legion::Copy& copy,
                        const CopyProfilingInfo& input) override;
  void select_sharding_functor(Legion::Mapping::MapperContext ctx,
                               const Legion::Copy& copy,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override;

  // Close mapping calls
  void select_close_sources(Legion::Mapping::MapperContext ctx,
                            const Legion::Close& close,
                            const SelectCloseSrcInput& input,
                            SelectCloseSrcOutput& output) override;
  void report_profiling(Legion::Mapping::MapperContext ctx,
                        const Legion::Close& close,
                        const CloseProfilingInfo& input) override;
  void select_sharding_functor(Legion::Mapping::MapperContext ctx,
                               const Legion::Close& close,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override;

  // Acquire mapping calls
  void map_acquire(Legion::Mapping::MapperContext ctx,
                   const Legion::Acquire& acquire,
                   const MapAcquireInput& input,
                   MapAcquireOutput& output) override;
  void report_profiling(Legion::Mapping::MapperContext ctx,
                        const Legion::Acquire& acquire,
                        const AcquireProfilingInfo& input) override;
  void select_sharding_functor(Legion::Mapping::MapperContext ctx,
                               const Legion::Acquire& acquire,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override;

  // Release mapping calls
  void map_release(Legion::Mapping::MapperContext ctx,
                   const Legion::Release& release,
                   const MapReleaseInput& input,
                   MapReleaseOutput& output) override;
  void select_release_sources(Legion::Mapping::MapperContext ctx,
                              const Legion::Release& release,
                              const SelectReleaseSrcInput& input,
                              SelectReleaseSrcOutput& output) override;
  void report_profiling(Legion::Mapping::MapperContext ctx,
                        const Legion::Release& release,
                        const ReleaseProfilingInfo& input) override;
  void select_sharding_functor(Legion::Mapping::MapperContext ctx,
                               const Legion::Release& release,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override;

  // Partition mapping calls
  void select_partition_projection(Legion::Mapping::MapperContext ctx,
                                   const Legion::Partition& partition,
                                   const SelectPartitionProjectionInput& input,
                                   SelectPartitionProjectionOutput& output) override;
  void map_partition(Legion::Mapping::MapperContext ctx,
                     const Legion::Partition& partition,
                     const MapPartitionInput& input,
                     MapPartitionOutput& output) override;
  void select_partition_sources(Legion::Mapping::MapperContext ctx,
                                const Legion::Partition& partition,
                                const SelectPartitionSrcInput& input,
                                SelectPartitionSrcOutput& output) override;
  void report_profiling(Legion::Mapping::MapperContext ctx,
                        const Legion::Partition& partition,
                        const PartitionProfilingInfo& input) override;
  void select_sharding_functor(Legion::Mapping::MapperContext ctx,
                               const Legion::Partition& partition,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override;

  // Fill mapper calls
  void select_sharding_functor(Legion::Mapping::MapperContext ctx,
                               const Legion::Fill& fill,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override;

  // Task execution mapping calls
  void configure_context(Legion::Mapping::MapperContext ctx,
                         const Legion::Task& task,
                         ContextConfigOutput& output) override;
  void map_future_map_reduction(Legion::Mapping::MapperContext ctx,
                                const FutureMapReductionInput& input,
                                FutureMapReductionOutput& output) override;
  void select_tunable_value(Legion::Mapping::MapperContext ctx,
                            const Legion::Task& task,
                            const SelectTunableInput& input,
                            SelectTunableOutput& output) override;

  // Must epoch mapping
  void select_sharding_functor(Legion::Mapping::MapperContext ctx,
                               const Legion::MustEpoch& epoch,
                               const SelectShardingFunctorInput& input,
                               MustEpochShardingFunctorOutput& output) override;
  void memoize_operation(Legion::Mapping::MapperContext ctx,
                         const Legion::Mappable& mappable,
                         const MemoizeInput& input,
                         MemoizeOutput& output) override;
  void map_must_epoch(Legion::Mapping::MapperContext ctx,
                      const MapMustEpochInput& input,
                      MapMustEpochOutput& output) override;

  // Dataflow graph mapping
  void map_dataflow_graph(Legion::Mapping::MapperContext ctx,
                          const MapDataflowGraphInput& input,
                          MapDataflowGraphOutput& output) override;

 private:
  /**
   * @brief Select streaming tasks to map.
   *
   * @param task The Legion task to potentially map.
   * @param stream_gen The task's streaming generation.
   * @param mapped_tasks The set of already mapped tasks. If this routine choses to map, it
   * will insert `task` into `mapped_tasks`.
   *
   * Conceptually we can model the execution of `N` tasks, each with `M` leafs as a `N x M`
   * matrix:
   *
   * |------|------|------|
   * | A[0] | A[1] | A[2] |
   * |------|------|------|
   * | B[0] | B[1] | A[2] |
   * |------|------|------|
   * | C[0] | C[1] | C[2] |
   * |------|------|------|
   *
   * Normally tasks execute "horizontally", where all instances of a particular task are
   * executed before any dependent operations. This is akin to executing the above matrix row
   * by row (assuming C depends on B depends on A).
   *
   * But in streaming, the tasks are executed "vertically", where for each submitted task, the
   * same slice of leaf tasks are executed. In the matrix example, this would be executing it
   * column by column.
   *
   * First, some terminology:
   *
   * - streaming generation: This is a unique ID that we assign to all tasks of a single
   *   streaming run with. In the example above, this is all tasks inside our matrix. If the
   *   streaming generation is the same, we are working on the same matrix.
   * - streaming size: This is the number of tasks per generation. In our case, the number of
   *   rows in the matrix.
   *
   * Prerequisites for the below:
   *
   * - All streaming tasks must have the same number of leaf tasks. In effect, each row of the
   *   matrix must have the same number of columns.
   *
   * - The sharding points for each task is the same. In effect, each shard should get a
   *   complete column of the matrix; if a node starts mapping column `c`, then it should
   *   expect to eventually see every row in that column.
   *
   * - All streaming runs must have a mapping fence inserted after they are scheduled. In
   *   effect, if we start working on streaming generation `N`, we expect to handle every task
   *   in the matrix before we start working on generation `N + 1`.
   *
   *   This restriction exists because the top-level code only knows how many tasks are part of
   *   the streaming generation (number of rows), but it doesn't yet know into how many shards
   *   those tasks will be parallelized into (number of columns).
   *
   *   A counter-example: suppose we did not have this restriction and the code below has just
   *   fully mapped a column. Some new tasks come in, but their streaming generation is
   *   different to our current generation. Does the mapper wait, hoping that more tasks from
   *   the previous generation show up? Or does it conclude that the previous generation has
   *   finished, and start mapping the new generation? Without the mapping fence, this question
   *   is undecidable. But with the mapping fence we know that if we encounter a new
   *   generation, that means the old generation is definitely finished.
   *
   * - All leaf tasks must have a unique `index_point`. This is similar to the first
   *   prerequisite, in that it guarantees all rows of the matrix have the same number of
   *   columns, because it says that for row `N`, it has `M` *unique* columns.
   */
  void select_streaming_tasks_to_map_(const Legion::Task& task,
                                      const legate::detail::StreamingGeneration& stream_gen,
                                      std::set<const Legion::Task*>* mapped_tasks);

 public:
  // Mapping control and stealing
  void select_tasks_to_map(Legion::Mapping::MapperContext ctx,
                           const SelectMappingInput& input,
                           SelectMappingOutput& output) override;
  void select_steal_targets(Legion::Mapping::MapperContext ctx,
                            const SelectStealingInput& input,
                            SelectStealingOutput& output) override;
  void permit_steal_request(Legion::Mapping::MapperContext ctx,
                            const StealRequestInput& input,
                            StealRequestOutput& output) override;

  // handling
  void handle_message(Legion::Mapping::MapperContext ctx, const MapperMessage& message) override;
  void handle_task_result(Legion::Mapping::MapperContext ctx,
                          const MapperTaskResult& result) override;
  void handle_instance_collection(Legion::Mapping::MapperContext ctx,
                                  const Legion::Mapping::PhysicalInstance& inst) override;

 private:
  using OutputMap = std::unordered_map<const Legion::RegionRequirement*,
                                       std::vector<Legion::Mapping::PhysicalInstance>*>;
  void map_legate_stores_(Legion::Mapping::MapperContext ctx,
                          const Legion::Mappable& mappable,
                          std::vector<std::unique_ptr<StoreMapping>>& mappings,
                          Processor target_proc,
                          OutputMap& output_map,
                          bool overdecomposed = false);
  void tighten_write_policies_(const Legion::Mappable& mappable,
                               const std::vector<std::unique_ptr<StoreMapping>>& mappings);
  [[nodiscard]] bool map_reduction_instance_(const Legion::Mapping::MapperContext& ctx,
                                             const Legion::Mappable& mappable,
                                             const Processor& target_proc,
                                             Legion::FieldID field,
                                             const std::vector<Legion::LogicalRegion>& regions,
                                             const InstanceMappingPolicy& policy,
                                             Memory target_memory,
                                             GlobalRedopID redop,
                                             Legion::LayoutConstraintSet* layout_constraints,
                                             Legion::Mapping::PhysicalInstance* result,
                                             bool* need_acquire,
                                             std::size_t* footprint);
  [[nodiscard]] bool map_regular_instance_(const Legion::Mapping::MapperContext& ctx,
                                           const Legion::Mappable& mappable,
                                           const std::set<const Legion::RegionRequirement*>& reqs,
                                           const InstanceMappingPolicy& policy,
                                           Legion::FieldID field,
                                           const Legion::LayoutConstraintSet& layout_constraints,
                                           Memory target_memory,
                                           bool must_alloc_collective_writes,
                                           std::vector<Legion::LogicalRegion>&& regions,
                                           Legion::Mapping::PhysicalInstance* result,
                                           bool* need_acquire,
                                           std::size_t* footprint);
  bool map_legate_store_(Legion::Mapping::MapperContext ctx,
                         const Legion::Mappable& mappable,
                         const StoreMapping& mapping,
                         const std::set<const Legion::RegionRequirement*>& reqs,
                         Processor target_proc,
                         Legion::Mapping::PhysicalInstance& result,
                         bool can_fail,
                         bool must_alloc_collective_writes);
  void report_failed_mapping_(Legion::Mapping::MapperContext ctx,
                              const Legion::Mappable& mappable,
                              const StoreMapping& mapping,
                              Memory target_memory,
                              GlobalRedopID redop,
                              std::size_t footprint);
  void legate_select_sources_(
    Legion::Mapping::MapperContext ctx,
    const Legion::Mapping::PhysicalInstance& target,
    const std::vector<Legion::Mapping::PhysicalInstance>& sources,
    const std::vector<Legion::Mapping::CollectiveView>& collective_sources,
    std::deque<Legion::Mapping::PhysicalInstance>& ranking);
  [[nodiscard]] static Legion::ShardingID find_mappable_sharding_functor_id_(
    const Legion::Mappable& mappable);

  [[nodiscard]] Legion::ShardingID find_sharding_functor_by_key_store_projection_(
    const std::vector<Legion::RegionRequirement>& requirements);

  [[nodiscard]] std::string_view retrieve_alloc_info_(Legion::Mapping::MapperContext ctx,
                                                      Legion::FieldSpace fs,
                                                      Legion::FieldID fid);

  Legion::Machine legion_machine_{Legion::Machine::get_machine()};
  Legion::Logger logger_{std::string{LOGGER_NAME}};

  using VariantCacheKey = std::pair<Legion::TaskID, Processor::Kind>;
  std::unordered_map<VariantCacheKey, std::optional<Legion::VariantID>, hasher<VariantCacheKey>>
    variants_{};

  InstanceManager local_instances_{};
  ReductionInstanceManager reduction_instances_{};

  std::unordered_map<Legion::Mapping::PhysicalInstance, std::string> creating_operation_{};
  LocalMachine local_machine_{};

  std::string mapper_name_{};

  // Streaming transformation related objects
  std::uint32_t streaming_current_gen_{};
  std::optional<DomainPoint> streaming_target_column_{};
  std::uint32_t streaming_rows_mapped_{};
  std::queue<Legion::Mapping::MapperEvent> deferral_events_{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/base_mapper.inl>
