/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/streaming.h>

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/discard.h>
#include <legate/operation/detail/mapping_fence.h>
#include <legate/operation/detail/operation.h>
#include <legate/operation/detail/task.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/legion_utilities.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/hash.h>

#include <algorithm>
#include <deque>
#include <functional>
#include <unordered_map>
#include <utility>

namespace legate::detail {

void StreamingGeneration::pack(BufferBuilder& buffer) const
{
  buffer.pack(generation);
  buffer.pack(size);
}

// ==========================================================================================

namespace {

/**
 * @brief Determine whether a task will be launched as a singleton task.
 *
 * Due to the fact that the launch domain is not known for AutoTask's before the task is
 * launched, this function can only definitively say whether a task will be a singleton task
 * for ManualTask's.
 *
 * @param task The task to check.
 *
 * @return `true` if the task will definitely be a singleton task, `false` if not.
 */
[[nodiscard]] bool is_singleton_task(const Task& task)
{
  if (task.kind() != Operation::Kind::MANUAL_TASK) {
    // Cannot yet predict launch domain of AutoTask.
    return false;
  }

  const auto& domain = static_cast<const ManualTask&>(task).launch_domain();

  return domain.get_dim() == 1 && domain.get_volume() == 1;
}

/**
 * @brief For a particular task, determine if argument was discarded and fixup the discard
 * privileges if so.
 *
 * If the task argument contains an argument which was discarded, this routine will mark that
 * particular discard operation as "handled".
 *
 * @param task_array_arg The argument to check.
 * @param discard_regions The set of discarded regions found for an ops stream.
 * @param discard_ops The list of discards which were handled by the current stream.
 */
void maybe_update_task_discards(
  TaskArrayArg* task_array_arg,
  std::unordered_map<std::pair<Legion::FieldID, Legion::LogicalRegion>, std::uint32_t, hasher<>>*
    discard_regions,
  SmallVector<std::uint32_t>* discard_ops)
{
  for (auto&& [store, _] : task_array_arg->mapping) {
    auto&& storage = store->get_storage();

    switch (storage->kind()) {
      // Ignore future and future-maps for now, it's not clear how these should be streamed (if
      // at all).
      case Storage::Kind::FUTURE: [[fallthrough]];
      case Storage::Kind::FUTURE_MAP: continue;
      case Storage::Kind::REGION_FIELD: break;
    }

    // Among other things, Legion does not yet support streaming for unbound stores. This is a
    // LEGATE_CHECK() because the code that computes the streaming section cut-points should
    // not have included this task in the streaming run to begin with.
    LEGATE_CHECK(!storage->unbound());

    auto&& rf     = storage->get_region_field();
    const auto it = discard_regions->find({rf->field_id(), rf->region()});

    if (it == discard_regions->end()) {
      // Not a discarded store, or we already handled it
      continue;
    }

    task_array_arg->privilege |= Legion::PrivilegeMode::LEGION_DISCARD_OUTPUT_MASK;

    discard_ops->emplace_back(it->second);
    discard_regions->erase(it);
    if (discard_regions->empty()) {
      return;
    }
  }
}

void scan_task_discards(
  Task* task,
  std::unordered_map<std::pair<Legion::FieldID, Legion::LogicalRegion>, std::uint32_t, hasher<>>*
    discard_regions,
  SmallVector<std::uint32_t>* discard_ops)
{
  // Handling reductions here is technically unsafe. See
  // https://github.com/nv-legate/legate.internal/pull/2338#discussion_r2191064347.
  for (auto&& task_args : {task->inputs(), task->outputs(), task->reductions()}) {
    for (auto&& arg : task_args) {
      // This is nasty, but I don't see a better way of doing this. We need to potentially
      // update the privilege on TaskArrayArg, but inputs() and outputs() return const
      // references.
      //
      // I don't want to expose mutable getters to them in Task just for the purposes of this
      // function (because they are sure to be abused).
      //
      // Adding a "Task::update_privilege()" of some description is also not great, since
      // that requires us to:
      //
      // 1. Tell Task what kind of argument we have, input, output etc (or have
      //    update_input_privilege(), update_output_privilege()).
      // 2. Tell Task which input/output we want to update. That requires either an integer
      //    index (so out of bounds errors possible), or some kind of iterator. In either
      //    case not exactly brilliant.
      auto& mut_arg = const_cast<TaskArrayArg&>(arg);

      maybe_update_task_discards(&mut_arg, discard_regions, discard_ops);
      if (discard_regions->empty()) {
        return;
      }
    }
  }
}

/**
 * @brief Find all of the discard operations in a particular operations stream and forward
 * propagate the discard privileges.
 *
 * @note We need to take by const std::vector<T>& instead of Span<const T> because we need to
 * return the iterators.
 *
 * @param ops The stream of operations to scan.
 *
 * @return discard_ops The discard operations that were handled directly within the ops stream.
 */
[[nodiscard]] std::pair<SmallVector<std::uint32_t>, std::uint32_t> forward_propagate_discards(
  const std::deque<InternalSharedPtr<Operation>>& ops)
{
  // Order of FieldID and LogicalRegion is deliberate. It is more expensive to compare
  // LogicalRegion's than it is to compare a single integer (FieldID).
  auto discard_regions = std::
    unordered_map<std::pair<Legion::FieldID, Legion::LogicalRegion>, std::uint32_t, hasher<>>{};
  // Can't use iterators for discard_ops because the standard says that std::deque::erase()
  // invalidates all iterators. So we must use indices instead.
  auto discard_ops = SmallVector<std::uint32_t>{};
  // Even during recursive flushes, there is at most only a single new generation being added.
  auto generation_size      = std::uint32_t{};
  auto finalized_generation = false;

  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    auto* const op = it->get();

    switch (op->kind()) {
      case Operation::Kind::ATTACH: [[fallthrough]];
      case Operation::Kind::COPY: [[fallthrough]];
      case Operation::Kind::EXECUTION_FENCE: [[fallthrough]];
      case Operation::Kind::FILL: [[fallthrough]];
      case Operation::Kind::GATHER: [[fallthrough]];
      case Operation::Kind::INDEX_ATTACH: [[fallthrough]];
      case Operation::Kind::MAPPING_FENCE: [[fallthrough]];
      case Operation::Kind::REDUCE: [[fallthrough]];
      case Operation::Kind::SCATTER: [[fallthrough]];
      case Operation::Kind::SCATTER_GATHER: [[fallthrough]];
      case Operation::Kind::TIMING: continue;
      case Operation::Kind::RELEASE_REGION_FIELD: {
        // TODO(jfaibussowit):
        // Should remove if store is unmapped
        continue;
      }

      case Operation::Kind::DISCARD: {
        const auto& discard = static_cast<const Discard&>(*op);

        discard_regions.emplace(std::make_pair(discard.field_id(), discard.region()),
                                std::distance(ops.begin(), std::next(it).base()));
        // We don't append to discard_ops here, because we only want to remove the discards that
        // this streaming run directly handles. I.e. if the streaming run is:
        //
        // - discard(A)
        // - foo(B, C)
        // - discard(B)
        //
        // Then we only want to remove discard(B) and leave discard(A) untouched.
        continue;
      }

        // Only user tasks need their privileges fixed
      case Operation::Kind::AUTO_TASK: [[fallthrough]];
      case Operation::Kind::MANUAL_TASK: break;
    }

    LEGATE_ASSERT(dynamic_cast<Task*>(op) != nullptr);

    auto* const task = static_cast<Task*>(op);

    // Don't count tasks that already have a generation assigned.
    if (task->streaming_generation().has_value()) {
      finalized_generation = true;
    } else {
      // Since the new generation could only have been appended, if we find any tasks without
      // an assigned generation between generations, then something has gone wrong. I.e. we
      // cannot have:
      //
      // 1. task(A) -> gen 1
      // 2. task(B) -> no gen
      // 3. task(C) -> gen 2
      //
      // It must always be
      //
      // 1. task(A) -> gen 1
      // 2. task(B) -> gen 2
      // 3. task(C) -> no gen
      LEGATE_CHECK(!finalized_generation);
      // Manual tasks whose launch domain is exactly {1} will be launched as singleton tasks
      // (not to be confused with index tasks whose launch volume is 1), and therefore won't be
      // considered part of a streaming generation.
      //
      // There is special handling for this in `BaseMapper::select_tasks_to_map()`. Any
      // singleton (non-index) tasks are immediately selected for mapping, and do not
      // participate in the usual vertical scheduling business.
      //
      // Therefore it is *absolutely necessary* that we do not bump the generation size
      // here. If we do, then any index launches in the same generation will be stuck waiting
      // for this task to schedule because it was counted as part of the generation size.
      if (!is_singleton_task(*task)) {
        ++generation_size;
      }
    }

    scan_task_discards(task, &discard_regions, &discard_ops);
  }

  return {std::move(discard_ops), generation_size};
}

/**
 * @brief Remove the handled discard operations from the operations queue.
 *
 * @param discard_ops The discard operations to remove.
 * @param ops The operations stream.
 */
void prune_handled_discard_ops(SmallVector<std::uint32_t> discard_ops,
                               std::deque<InternalSharedPtr<Operation>>* ops)
{
  // Note, we need to remove the discard operations starting from the back because otherwise
  // the other indices pointed to by discard_ops get invalidated. I.e. we must remove item 3
  // before we remove items 2 and 1.
  std::sort(discard_ops.begin(), discard_ops.end(), std::greater<>{});
  for (auto to_remove_idx : discard_ops) {
    const auto it = std::next(ops->begin(), to_remove_idx);

    LEGATE_ASSERT(it < ops->end());
    LEGATE_ASSERT(dynamic_cast<Discard*>(it->get()) != nullptr);
    ops->erase(it);
  }
}

/**
 * @brief Assign the streaming generation information to the newly added streaming run.
 *
 * @param generation_size The number of operations comprising the new generation.
 * @param ops The operations stream.
 */
void assign_streaming_generation(std::uint32_t generation_size,
                                 std::deque<InternalSharedPtr<Operation>>* ops)
{
  static auto streaming_generation = std::uint32_t{0};
  auto last_set_it                 = ops->rbegin();

  ++streaming_generation;
  // We know that any new streaming generations, by virtue of being new, must only have been
  // *appended* to the operations queue. So we can search the container backwards.
  for (auto rit = ops->rbegin(); rit != ops->rend(); ++rit) {
    auto* const task = dynamic_cast<Task*>(rit->get());

    if (!task) {
      continue;
    }

    if (task->streaming_generation().has_value()) {
      // In the case of a recursive scheduling flush, we might already have assigned this
      // task a particular streaming generation. We must not change that generation because
      // those tasks must still map with their previous generation. Otherwise, the mapper
      // will loop forever trying to finish mapping a generation which never fully arrives
      // (well, it does, but now it has the next generation).
      //
      // Instead, if we detect this case, we need to make sure there is a mapping (or
      // execution fence) in between these tasks, to ensure that each maps separately.
      const auto has_fence =
        std::any_of(last_set_it, rit, [](const InternalSharedPtr<Operation>& op) {
          const auto kind = op->kind();

          switch (kind) {
            case Operation::Kind::ATTACH: [[fallthrough]];
            case Operation::Kind::COPY: [[fallthrough]];
            case Operation::Kind::DISCARD: [[fallthrough]];
            case Operation::Kind::FILL: [[fallthrough]];
            case Operation::Kind::GATHER: [[fallthrough]];
            case Operation::Kind::INDEX_ATTACH: [[fallthrough]];
            case Operation::Kind::REDUCE: [[fallthrough]];
            case Operation::Kind::SCATTER: [[fallthrough]];
            case Operation::Kind::SCATTER_GATHER: [[fallthrough]];
            case Operation::Kind::TIMING: [[fallthrough]];
            case Operation::Kind::AUTO_TASK: [[fallthrough]];
            case Operation::Kind::MANUAL_TASK: [[fallthrough]];
            case Operation::Kind::RELEASE_REGION_FIELD: {
              return false;
            }

            case Operation::Kind::EXECUTION_FENCE: [[fallthrough]];
            case Operation::Kind::MAPPING_FENCE: {
              return true;
            }
          }
          LEGATE_ABORT("Unhandled operation kind ", to_underlying(kind));
          return false;
        });

      if (!has_fence) {
        // Insert a mapping fence into the task stream. Do not go through the usual
        // `Runtime::issue_mapping_fence()` since that will append it (and given we are working
        // on a generic pointer to ops queue, may not even append it to this one -- though
        // exceedingly unlikely).
        ops->insert(std::next(last_set_it).base(),
                    make_internal_shared<MappingFence>(Runtime::get_runtime().new_op_id()));
      }
      break;
    }

    if (is_singleton_task(*task)) {
      // Singleton tasks don't participate in streaming generation analysis. See
      // `BaseMapper::select_tasks_to_map()` for further discussion.
      continue;
    }

    last_set_it = rit;
    task->set_streaming_generation({{streaming_generation, generation_size}});
  }
}

}  // namespace

void process_streaming_run(std::deque<InternalSharedPtr<Operation>>* ops)
{
  auto [discard_ops, generation_sizes] = forward_propagate_discards(*ops);

  prune_handled_discard_ops(std::move(discard_ops), ops);
  assign_streaming_generation(generation_sizes, ops);
}

}  // namespace legate::detail
