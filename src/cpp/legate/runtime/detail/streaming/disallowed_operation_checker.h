/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/streaming/base_operation_checker.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <string_view>

namespace legate::detail {

class Operation;
class Strategy;
class StreamingErrorContext;

/**
 * @brief Checks if the Operation is not allowed inside a Streaming Scope.
 */
class DisallowedOp final : public BaseOperationChecker {
 public:
  /**
   * @return true if the operation does not support streaming.
   */
  [[nodiscard]] bool is_streamable(const InternalSharedPtr<Operation>& op,
                                   const std::optional<InternalSharedPtr<Strategy>>& strategy,
                                   StreamingErrorContext* ctx) override;
  /**
   * @brief Returns name for printing errors.
   */
  [[nodiscard]] std::string_view name() const override;
};

}  // namespace legate::detail
