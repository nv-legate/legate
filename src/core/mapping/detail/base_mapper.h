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

#include "core/data/scalar.h"
#include "core/mapping/detail/machine.h"
#include "core/mapping/detail/mapping.h"
#include "core/runtime/detail/library.h"
#include "core/utilities/detail/hash.h"
#include "core/utilities/typedefs.h"

#include "legion.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace legate::mapping::detail {

class InstanceManager;
class ReductionInstanceManager;

class BaseMapper final : public Legion::Mapping::Mapper, public MachineQueryInterface {
 public:
  BaseMapper(mapping::Mapper* legate_mapper,
             Legion::Mapping::MapperRuntime* mapper_runtime,
             const legate::detail::Library* library);

  ~BaseMapper() override;

  BaseMapper(const BaseMapper& rhs)            = delete;
  BaseMapper& operator=(const BaseMapper& rhs) = delete;

 protected:
  // Start-up methods
  [[nodiscard]] std::string create_logger_name() const;

 public:
  // MachineQueryInterface
  [[nodiscard]] const std::vector<Processor>& cpus() const override;
  [[nodiscard]] const std::vector<Processor>& gpus() const override;
  [[nodiscard]] const std::vector<Processor>& omps() const override;
  [[nodiscard]] uint32_t total_nodes() const override;
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
  void map_replicate_task(Legion::Mapping::MapperContext ctx,
                          const Legion::Task& task,
                          const MapTaskInput& input,
                          const MapTaskOutput& default_output,
                          MapReplicateTaskOutput& output) override;
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

 protected:
  using OutputMap = std::unordered_map<const Legion::RegionRequirement*,
                                       std::vector<Legion::Mapping::PhysicalInstance>*>;
  void map_legate_stores(Legion::Mapping::MapperContext ctx,
                         const Legion::Mappable& mappable,
                         std::vector<std::unique_ptr<StoreMapping>>& mappings,
                         Processor target_proc,
                         OutputMap& output_map);
  void tighten_write_policies(const Legion::Mappable& mappable,
                              std::vector<std::unique_ptr<StoreMapping>>& mappings);
  bool map_legate_store(Legion::Mapping::MapperContext ctx,
                        const Legion::Mappable& mappable,
                        const StoreMapping& mapping,
                        const std::set<const Legion::RegionRequirement*>& reqs,
                        Processor target_proc,
                        Legion::Mapping::PhysicalInstance& result,
                        bool can_fail);
  void report_failed_mapping(const Legion::Mappable& mappable,
                             unsigned index,
                             Memory target_memory,
                             Legion::ReductionOpID redop,
                             size_t footprint);
  void legate_select_sources(Legion::Mapping::MapperContext ctx,
                             const Legion::Mapping::PhysicalInstance& target,
                             const std::vector<Legion::Mapping::PhysicalInstance>& sources,
                             const std::vector<Legion::Mapping::CollectiveView>& collective_sources,
                             std::deque<Legion::Mapping::PhysicalInstance>& ranking);
  static Legion::ShardingID find_mappable_sharding_functor_id(const Legion::Mappable& mappable);

  bool has_variant(Legion::Mapping::MapperContext ctx, const Legion::Task& task, TaskTarget target);
  std::optional<Legion::VariantID> find_variant(Legion::Mapping::MapperContext ctx,
                                                const Legion::Task& task,
                                                Processor::Kind kind);

  Legion::ShardingID find_sharding_functor_by_key_store_projection(
    const std::vector<Legion::RegionRequirement>& requirements);

 private:
  mapping::Mapper* legate_mapper_{};

 public:
  Legion::Mapping::MapperRuntime* const mapper_runtime{};
  const Legion::Machine legion_machine;
  const legate::detail::Library* library{};
  Legion::Logger logger;

 private:
  std::string mapper_name{};

 protected:
  using VariantCacheKey = std::pair<Legion::TaskID, Processor::Kind>;
  std::unordered_map<VariantCacheKey, std::optional<Legion::VariantID>, hasher<VariantCacheKey>>
    variants{};

  InstanceManager* local_instances{};
  ReductionInstanceManager* reduction_instances{};
  LocalMachine local_machine{};
};

}  // namespace legate::mapping::detail

#include "core/mapping/detail/base_mapper.inl"
