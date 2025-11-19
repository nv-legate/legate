/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/streaming/disallowed_operation_checker.h>

#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/strategy.h>
#include <legate/runtime/detail/streaming/util.h>
#include <legate/utilities/detail/formatters.h>

namespace legate::detail {

std::string_view DisallowedOp::name() const { return "Disallowed Operation"; }

bool DisallowedOp::is_streamable(const InternalSharedPtr<Operation>& op,
                                 const std::optional<InternalSharedPtr<Strategy>>&,
                                 StreamingErrorContext* ctx)
{
  if (op->needs_flush() || (!op->supports_streaming())) {
    ctx->append("Operation {} does not support streaming", *op);
    return false;
  };

  return true;
}

}  // namespace legate::detail
