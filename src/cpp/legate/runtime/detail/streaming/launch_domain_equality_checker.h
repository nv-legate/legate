/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/streaming/base_operation_checker.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <optional>
#include <string_view>

namespace legate::detail {

class Operation;
class Strategy;
class StreamingErrorContext;

/**
 * @brief Checks that all Operations inside a Streaming Scope have equal Launch Domains.
 */
class LaunchDomainEquality final : public BaseOperationChecker {
 public:
  /**
   * @return true if the operation has no launch domain or has a launch domain
   * equal to the previous operations.
   */
  [[nodiscard]] bool is_streamable(const InternalSharedPtr<Operation>& op,
                                   const std::optional<InternalSharedPtr<Strategy>>& strategy,
                                   StreamingErrorContext* ctx) override;
  /**
   * @brief Returns name for printing errors.
   */
  [[nodiscard]] std::string_view name() const override;

 private:
  std::optional<Domain> launch_domain_{std::nullopt};
};

}  // namespace legate::detail
