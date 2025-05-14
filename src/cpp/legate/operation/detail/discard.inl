/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/discard.h>

namespace legate::detail {

inline Discard::Discard(std::uint64_t unique_id,
                        Legion::LogicalRegion region,
                        Legion::FieldID field_id)
  : Operation{unique_id}, region_{std::move(region)}, field_id_{field_id}
{
}

inline Operation::Kind Discard::kind() const { return Kind::DISCARD; }

inline bool Discard::supports_streaming() const { return true; }

}  // namespace legate::detail
