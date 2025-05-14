/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/tuning/parallel_policy.h>

#include <cstdint>

namespace legate {

ParallelPolicy& ParallelPolicy::with_streaming(bool streaming)
{
  streaming_ = streaming;
  return *this;
}

ParallelPolicy& ParallelPolicy::with_overdecompose_factor(std::uint32_t overdecompose_factor)
{
  overdecompose_factor_ = overdecompose_factor;
  return *this;
}

bool ParallelPolicy::operator==(const ParallelPolicy& other) const
{
  return streaming() == other.streaming() && overdecompose_factor() == other.overdecompose_factor();
}

bool ParallelPolicy::operator!=(const ParallelPolicy& other) const { return !(*this == other); }

}  // namespace legate
