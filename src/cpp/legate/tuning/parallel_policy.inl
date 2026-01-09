/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/tuning/parallel_policy.h>

namespace legate {

inline bool ParallelPolicy::streaming() const { return streaming_mode() != StreamingMode::OFF; }

inline StreamingMode ParallelPolicy::streaming_mode() const { return streaming_mode_; }

inline std::uint32_t ParallelPolicy::overdecompose_factor() const { return overdecompose_factor_; }

}  // namespace legate
