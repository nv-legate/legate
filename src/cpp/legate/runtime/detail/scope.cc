/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/scope.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/formatters.h>

#include <utility>

namespace legate::detail {
namespace {

/**
 * @brief Helper function to throw an exception and clear the runtime's scheduling
 * window.
 *
 * @param msg Exception message.
 */
[[noreturn]] void flush_window_and_throw(const char* const msg)
{
  auto&& rt = Runtime::get_runtime();
  rt.clear_scheduling_window(Runtime::PrivateKey{});
  throw TracedException<std::invalid_argument>{msg};
}

/**
 * @brief Check for nested scopes with different parallel policies and throw an
 * exception if the nesting is not valid.
 *
 * @param outer_policy Outer scope's ParallelPolicy
 * @param inner_policy Inner scope's ParallelPolicy
 *
 * Following rules apply when outer policy is different from inner
 *
 * Outer Scope Mode | Inner Scope Mode | Behavior
 * NONE             | RELAXED or STRICT| mapping fence + flush
 * RELAXED          | STRICT           | mapping fence + flush
 * RELAXED          | NONE             | throw exception
 * STRICT           | RELAXED or NONE  | throw exception
 * STRICT           | STRICT           | throw exception because something else in
 *                                       ParallelPolicy changed, e.g., OD factor
 */
void check_policy_change_at_scope_begin(const ParallelPolicy& outer_policy,
                                        const ParallelPolicy& inner_policy)
{
  // precondition
  LEGATE_ASSERT(outer_policy != inner_policy);

  const char* const inner_off_err =
    "cannot nest a non-streaming scope inside a STRICT streaming scope";

  const char* const inner_relaxed_err =
    "cannot nest a RELAXED streaming scope inside a STRICT streaming scope";

  const char* const inner_diff_err =
    "cannot change the parallel policy when nesting a scope inside a STRICT streaming scope";

  switch (outer_policy.streaming_mode()) {
    case legate::StreamingMode::OFF: [[fallthrough]];
    case legate::StreamingMode::RELAXED: return;  // no need to check inner.
    case legate::StreamingMode::STRICT:
      switch (inner_policy.streaming_mode()) {
        case legate::StreamingMode::OFF: flush_window_and_throw(inner_off_err);
        case legate::StreamingMode::RELAXED: flush_window_and_throw(inner_relaxed_err);
        case legate::StreamingMode::STRICT:
          flush_window_and_throw(inner_diff_err);  // error because inner_policy != outer_policy
      }
      LEGATE_ABORT(
        fmt::format("Invalid inner_policy streaming mode: {}", inner_policy.streaming_mode()));
  }

  LEGATE_ABORT(
    fmt::format("Invalid outer_policy streaming mode: {}", outer_policy.streaming_mode()));
}

}  // namespace

void Scope::trigger_exchange_side_effects(const ParallelPolicy& new_policy,
                                          Scope::ChangeKind change_kind) const
{
  // TODO(amberhassaan): The equality check below can be relaxed to a compatibility
  // check in the future because we may add more attributes and flags to
  // ParallelPolicy that may not affect Streaming and may not need to flush or
  // issue a mapping fence.
  if (new_policy != parallel_policy_) {
    auto&& rt = Runtime::get_runtime();

    switch (change_kind) {
      case ChangeKind::SCOPE_BEG:
        check_policy_change_at_scope_begin(parallel_policy_, new_policy);
        break;
      case ChangeKind::SCOPE_END: break;
    }

    if (new_policy.streaming() || parallel_policy_.streaming()) {
      // Note, we want to issue this mapping fence if either the incoming or outgoing scope are
      // streaming, because during scheduling window flushes, the streaming generation will
      // change. See discussion in `BaseMapping::select_streaming_tasks_to_map()` for why this
      // fence is needed.
      rt.issue_mapping_fence();
      // In a multi-rank run when we're inside a streaming scope we stop consensus matching
      // discarded fields as that ends up flushing the current scheduling window and
      // waits on existing matches. This results in breaking the streaming window. We issue
      // a match at the end of a streaming scope in case there are outstanding matches
      // that might have triggered inside the scope.
      rt.issue_field_match();
    }
    rt.flush_scheduling_window(/*streaming_scope_change=*/true);
  }
}

ParallelPolicy Scope::exchange_parallel_policy(ParallelPolicy new_policy)
{
  return std::exchange(parallel_policy_, std::move(new_policy));
}

}  // namespace legate::detail
