/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/streaming/dependency_checker.h>

#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/partitioning/detail/partition/tiling.h>
#include <legate/partitioning/detail/strategy.h>
#include <legate/runtime/detail/streaming/util.h>
#include <legate/utilities/detail/formatters.h>

#include <fmt/ranges.h>

#include <algorithm>

namespace legate::detail {

namespace {

/**
 * @brief Compare two tiling partitions.
 *
 * @param this_tiling First tiling partition.
 * @param that_tiling Second tiling partition.
 *
 * @return true if equal
 */
[[nodiscard]] bool is_equal_tiling(const Tiling& this_tiling, const Tiling& that_tiling)
{
  if (log_streaming().want_debug()) {
    log_streaming().debug() << fmt::format("Comparing last tile {}, {}, curr tile {}, {}\n",
                                           fmt::join(this_tiling.tile_shape(), ", "),
                                           fmt::join(this_tiling.color_shape(), ", "),
                                           fmt::join(that_tiling.tile_shape(), ", "),
                                           fmt::join(that_tiling.color_shape(), ", "));
  }

  const auto compare_spans = [](const auto& s1, const auto& s2) {
    return std::equal(s1.begin(), s1.end(), s2.begin(), s2.end());
  };

  return compare_spans(this_tiling.tile_shape(), that_tiling.tile_shape()) &&
         compare_spans(this_tiling.color_shape(), that_tiling.color_shape());
}

/**
 * @brief retrieve the partitioning of a store from a strategy.
 *
 * @param strategy The partitioning strategy.
 * @param var Variable assigned to the store.
 * @param store The logical store.
 *
 * @return shared pointer to partitioning of the store.
 */
[[nodiscard]] InternalSharedPtr<Partition> get_partition(
  const Strategy& strategy, const Variable* var, const InternalSharedPtr<LogicalStore>& store)
{
  if (strategy.has_assignment(*var)) {
    return strategy[*var];
  }

  if (const auto& maybe_kp = store->get_current_key_partition(); maybe_kp.has_value()) {
    return maybe_kp.value();
  }

  return create_no_partition();
}

}  // namespace

std::string_view DependencyChecker::name() const { return "Dependency Checker"; }

// TODO(amberhassaan): make this a method on Partition class hierarchy. Tracked via
// https://github.com/nv-legate/legate.internal/issues/3391
bool DependencyChecker::have_equal_partitioning_(const InternalSharedPtr<LogicalStore>& store,
                                                 const AccessInfo& last,
                                                 const AccessInfo& curr)
{
  if (log_streaming().want_debug()) {
    log_streaming().debug() << fmt::format(
      "Comparing partitioning of store {}, for task {} (partition : {}) and task {} (partition "
      ": {}",
      *store,
      *last.op,
      last.partition->to_string(),
      *curr.op,
      curr.partition->to_string());
  }

  auto get_type = [](const auto& ptr) -> const std::type_info& {
    const auto& r = *ptr;
    return typeid(r);
  };
  const auto& last_type = get_type(last.partition);
  const auto& curr_type = get_type(curr.partition);

  // can't compare partitions of different types
  if (last_type != curr_type) {
    return false;
  }

  if (last_type == typeid(NoPartition)) {
    return true;  // equally NO_PARTITION for both
  }

  if (last_type == typeid(Tiling)) {
    return is_equal_tiling(static_cast<const Tiling&>(*(last.partition)),
                           static_cast<const Tiling&>(*(curr.partition)));
  }

  return false;  // TODO(amberhassaan): add support for image and opaque partitions.
                 // Image maybe expensive due to its data dependent nature
}

std::optional<DependencyChecker::AccessInfo>& DependencyChecker::get_last_access_(
  const InternalSharedPtr<LogicalStore>& store)
{
  const auto& root_storage = store->get_storage()->get_root();
  return per_store_accesses_[root_storage->id()];
}

bool DependencyChecker::fail_with_msg_(const InternalSharedPtr<LogicalStore>& store,
                                       const AccessInfo& curr,
                                       const AccessInfo& last,
                                       std::string_view reason,
                                       StreamingErrorContext* ctx)
{
  ctx->append(
    /*fmt_str=*/
    "Failed Dependency Check due to {} for store {},"
    " accessed by operation {}, last accessed by operation {}",
    reason,
    *store,
    *curr.op,
    *last.op);
  return false;
}

bool DependencyChecker::analyze_inputs_(const InternalSharedPtr<Operation>& op,
                                        const Strategy& strategy,
                                        StreamingErrorContext* ctx)
{
  for (const auto& [store, var] : op->input_stores()) {
    auto part    = get_partition(strategy, var, store);
    auto in_info = AccessInfo{op, part, AccessMode::READ};

    auto& maybe_last_access = get_last_access_(store);

    if (maybe_last_access.has_value()) {
      const auto& last_access = maybe_last_access.value();

      switch (last_access.access_mode) {
        // NOTE(amberhassaan): In theory, multiple reads to the same
        // store with different partitionings are streamable if there are no writes
        // or reductions to that store. However, we expect this to be rare
        // and allowing it forces us to check writes and reductions against every
        // previous read instead of just the last access. Allowing such
        // non-pointwise reads also affect the memory savings expected from
        // streaming and other optimizations, e.g., operation fusion.
        case AccessMode::READ:  // Read after Read
          if (!have_equal_partitioning_(store, last_access, in_info)) {
            return fail_with_msg_(
              store, in_info, last_access, "Multiple Reads with unequal partitioning", ctx);
          }
          break;
        case AccessMode::WRITE:  // Read after Write
          if (!have_equal_partitioning_(store, last_access, in_info)) {
            return fail_with_msg_(
              store, in_info, last_access, "RAW dependency, unequal partitioning", ctx);
          }
          break;
        case AccessMode::REDUCE:  // Read after Reduce
          if (store->volume() == 1) {
            if (const auto& ld = strategy.launch_domain(*op); ld.get_volume() == 1) {
              // This is a hack to allow a singleton task (specifically HDF5Write)
              // to pass this check, where we encode a control dependency using a
              // dummy data dependency with a store that is reduced by the prior
              // task and read by the current task. No actual reduction on the
              // store performed but reduction is needed to encode a dependency on
              // all the leaf tasks of the prior task.
              continue;
            }
          }
          return fail_with_msg_(store, in_info, last_access, "reading a store being reduced", ctx);
      }
    }
    maybe_last_access = in_info;
  }

  return true;
}

bool DependencyChecker::analyze_outputs_(const InternalSharedPtr<Operation>& op,
                                         const Strategy& strategy,
                                         StreamingErrorContext* ctx)
{
  for (const auto& [store, var] : op->output_stores()) {
    auto part     = get_partition(strategy, var, store);
    auto out_info = AccessInfo{op, part, AccessMode::WRITE};

    auto& maybe_last_access = get_last_access_(store);

    if (maybe_last_access.has_value()) {
      const auto& last_access = maybe_last_access.value();

      switch (last_access.access_mode) {
        case AccessMode::READ:  // Write after Read
          if (!have_equal_partitioning_(store, last_access, out_info)) {
            return fail_with_msg_(
              store, out_info, last_access, "WAR dependency, unequal partitioning", ctx);
          }
          break;
        case AccessMode::WRITE:  // Write after Write
          if (!have_equal_partitioning_(store, last_access, out_info)) {
            return fail_with_msg_(
              store, out_info, last_access, "WAW dependency, unequal partitioning", ctx);
          }
          break;
        case AccessMode::REDUCE:  // Write after Reduce
          return fail_with_msg_(store, out_info, last_access, "writing a store being reduced", ctx);
      }
    }
    maybe_last_access = out_info;
  }

  return true;
}

bool DependencyChecker::analyze_reductions_(const InternalSharedPtr<Operation>& op,
                                            const Strategy& strategy,
                                            StreamingErrorContext* ctx)
{
  for (const auto& [store, var] : op->reduction_stores()) {
    auto part     = get_partition(strategy, var, store);
    auto red_info = AccessInfo{op, part, AccessMode::REDUCE};

    auto& maybe_last_access = get_last_access_(store);

    if (maybe_last_access.has_value()) {
      const auto& last_access = maybe_last_access.value();

      switch (last_access.access_mode) {
        case AccessMode::READ:  // Reduce after Read
          if (!have_equal_partitioning_(store, last_access, red_info)) {
            return fail_with_msg_(store,
                                  red_info,
                                  last_access,
                                  "non pointwise dependency between "
                                  "reduction and a previous read",
                                  ctx);
          }
          break;
        case AccessMode::WRITE:  // Reduce after Write
          if (!have_equal_partitioning_(store, last_access, red_info)) {
            return fail_with_msg_(store,
                                  red_info,
                                  last_access,
                                  "non pointwise dependency between "
                                  "reduction and a previous write",
                                  ctx);
          }
          break;
        case AccessMode::REDUCE:  // Reduce after Reduce
          return fail_with_msg_(
            store, red_info, last_access, "multiple reductions on the store", ctx);
      }
    }
    maybe_last_access = red_info;
  }

  return true;
}

bool DependencyChecker::is_streamable(
  const InternalSharedPtr<Operation>& op,
  const std::optional<InternalSharedPtr<Strategy>>& maybe_strategy,
  StreamingErrorContext* ctx)
{
  if (!maybe_strategy.has_value()) {
    // internal ops that don't need partitioning except ManualTask
    return true;
  }

  const Strategy& strategy = *maybe_strategy.value();

  return analyze_inputs_(op, strategy, ctx) && analyze_outputs_(op, strategy, ctx) &&
         analyze_reductions_(op, strategy, ctx);
}

}  // namespace legate::detail
