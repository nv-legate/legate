/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/mapping_fence.h>

namespace legate::detail {

inline MappingFence::MappingFence(std::uint64_t unique_id) : Operation{unique_id} {}

inline Operation::Kind MappingFence::kind() const { return Kind::MAPPING_FENCE; }

inline bool MappingFence::needs_flush() const { return false; }

inline bool MappingFence::needs_partitioning() const { return false; }

}  // namespace legate::detail
