/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll_comm.h>

namespace legate::detail::comm::coll {

/**
 * @brief Apply a reduction operation for each index in destination and source buffer. Store result
 * in destination buffer.
 *
 * @tparam T The data type of the buffers.
 * @param dst Destination buffer (also serves as one input, modified in-place).
 * @param src Source buffer.
 * @param count Number of elements.
 * @param op Reduction operation to apply.
 */
template <typename T>
void apply_reduction_typed(void* dst, const void* src, unsigned count, ReductionOpKind op);

}  // namespace legate::detail::comm::coll

#include <legate/comm/detail/reduction_helpers.inl>
