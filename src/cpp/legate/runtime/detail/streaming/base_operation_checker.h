/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/internal_shared_ptr.h>

#include <optional>
#include <string_view>

namespace legate::detail {

class Operation;
class Strategy;
class StreamingErrorContext;

/**
 * @brief virtual base class for checking operations individually.
 */
class BaseOperationChecker {
 public:
  /**
   * @brief Derived classes check each op but also can accumulate some state at the same
   * time about the history of ops seen so far in the scheduling window.
   *
   * @param op Operation.
   * @param strategy Optional Strategy.
   * @param ctx An object to collect helpful error messages describing why the
   * is_streamable test failed.
   *
   * @return true if the operation is streamable. The return value may also depend
   * on internal state accumulated from the previous calls to this function.
   */
  [[nodiscard]] virtual bool is_streamable(
    const InternalSharedPtr<Operation>& op,
    const std::optional<InternalSharedPtr<Strategy>>& strategy,
    StreamingErrorContext* ctx) = 0;

  /**
   * @brief Returns name for printing errors.
   */
  [[nodiscard]] virtual std::string_view name() const = 0;

  virtual ~BaseOperationChecker() = default;
};

}  // namespace legate::detail
