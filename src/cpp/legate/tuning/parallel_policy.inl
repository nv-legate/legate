/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/tuning/parallel_policy.h>

namespace legate {

inline bool ParallelPolicy::streaming() const { return streaming_; }

inline std::uint32_t ParallelPolicy::overdecompose_factor() const { return overdecompose_factor_; }

}  // namespace legate
