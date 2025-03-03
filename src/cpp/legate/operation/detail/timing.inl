/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/timing.h>

namespace legate::detail {

inline Timing::Timing(std::uint64_t unique_id,
                      Precision precision,
                      InternalSharedPtr<LogicalStore> store)
  : Operation{unique_id}, precision_{precision}, store_{std::move(store)}
{
}

inline Operation::Kind Timing::kind() const { return Kind::TIMING; }

}  // namespace legate::detail
