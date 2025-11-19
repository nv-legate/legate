/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/runtime/detail/streaming/util.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <deque>

namespace legate::detail {

class Operation;

/**
 * @brief Given a queue of Operations, find a streamable prefix of the queue by
 * running some checks on the queue.
 *
 * @param in_window Operations from the schedulign window.
 *
 * @return deque containing the prefix of operations from `ops` that can be
 * streamed together.
 *
 * @throws std::invalid_argument if the window of operations is not streamable and
 * the user has specified streaming mode as `StreamingMode::STRICT`.
 */
[[nodiscard]] std::deque<InternalSharedPtr<Operation>> extract_streamable_prefix(
  std::deque<InternalSharedPtr<Operation>>* in_window);

/**
 * @brief Process a streaming section.
 *
 * During a streaming section, we want to:
 *
 * 1. Find each discard operation and note the store...
 * 2. Find the last task (within the stream) which used the discarded store, and...
 * 3. Mark the store for that particular task with `LEGION_DISCARD_OUTPUT_MASK`, so that Legion
 *    knows it can eargerly collect the memory of the store immediately after the task
 *    finishes.
 *
 * Together, these transformations have the effect of "moving" the discarding of a store
 * "forward" in the task stream. This makes the eventual discard operation pointless, because
 * the store should already have been eagerly collected after the last using task ran. So we
 * remove the discard operation (but crucially *only* the discard ops that were handled in the
 * ops stream).
 *
 * The final step is to inform every task in the streaming section that they are a streaming
 * task. This information is needed by the mapper (in `select_tasks_to_map()`) in order to
 * properly vertically schedule the tasks.
 *
 * @param ops The operations stream to scan.
 */
void process_streaming_run(std::deque<InternalSharedPtr<Operation>>* ops);

}  // namespace legate::detail
