/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/inline_allocation.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

/**
 * @brief A task-local temporary buffer.
 *
 * A `TaskLocalBuffer` is, as the name implies, "local" to a task. Its lifetime is bound to
 * that of the task. When the task ends, the buffer is destroyed. It is most commonly used as
 * temporary scratch-space within tasks for that reason.
 *
 * The buffer is allocated immediately at the point when `TaskLocalBuffer` is created, so it is
 * safe to use it immediately, even if it used asynchronously (for example, in GPU kernel
 * launches) after the fact.
 */
class TaskLocalBuffer {
 public:
  /**
   * @brief Construct a `TaskLocalBuffer`.
   *
   * @param buf The Legion buffer from which to construct this buffer from.
   * @param type The type to interpret `buf` as.
   * @param bounds The extents of the buffer.
   */
  TaskLocalBuffer(const Legion::UntypedDeferredBuffer<>& buf,
                  InternalSharedPtr<Type> type,
                  const Domain& bounds);

  /**
   * @return The type of the buffer.
   */
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;

  /**
   * @return The dimension of the buffer
   */
  [[nodiscard]] std::int32_t dim() const;

  /**
   * @return The shape of the buffer.
   */
  [[nodiscard]] const Domain& domain() const;

  /**
   * @return The memory kind of the buffer.
   */
  [[nodiscard]] mapping::StoreTarget memory_kind() const;

  /**
   * @return The legion buffer handle.
   */
  [[nodiscard]] const Legion::UntypedDeferredBuffer<>& legion_buffer() const;

  /**
   * @brief Get the `InlineAllocation` for the buffer.
   *
   * This routine constructs a fresh `InlineAllocation` for each call. This process may not be
   * cheap, so the user is encouraged to call this sparingly.
   *
   * @return The inline allocation object.
   */
  [[nodiscard]] InlineAllocation get_inline_allocation() const;

 private:
  Legion::UntypedDeferredBuffer<> buf_{};
  InternalSharedPtr<Type> type_{};
  Domain domain_{};
};

}  // namespace legate::detail

#include <legate/data/detail/buffer.inl>
