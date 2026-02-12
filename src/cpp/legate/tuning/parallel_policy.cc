/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/tuning/parallel_policy.h>

#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/formatters.h>

#include <fmt/format.h>

#include <cstdint>

namespace legate {

ParallelPolicy::ParallelPolicy(const detail::Config& config, PrivateKey)
  : cpu_partitioning_threshold_{config.min_cpu_chunk()},
    gpu_partitioning_threshold_{config.min_gpu_chunk()},
    omp_partitioning_threshold_{config.min_omp_chunk()}
{
}

ParallelPolicy::ParallelPolicy()
  : ParallelPolicy{detail::Runtime::get_runtime().config(), PrivateKey{}}
{
}

ParallelPolicy& ParallelPolicy::with_streaming(StreamingMode mode)
{
  streaming_mode_ = mode;
  return *this;
}

ParallelPolicy& ParallelPolicy::with_overdecompose_factor(std::uint32_t overdecompose_factor)
{
  overdecompose_factor_ = overdecompose_factor;
  return *this;
}

ParallelPolicy& ParallelPolicy::with_partitioning_threshold(mapping::TaskTarget target,
                                                            std::uint64_t threshold)
{
  switch (target) {
    case mapping::TaskTarget::CPU: cpu_partitioning_threshold_ = threshold; return *this;
    case mapping::TaskTarget::GPU: gpu_partitioning_threshold_ = threshold; return *this;
    case mapping::TaskTarget::OMP: omp_partitioning_threshold_ = threshold; return *this;
  }
  LEGATE_ABORT(fmt::format("Invalid target option: {}", detail::to_underlying(target)));
  return *this;
}

std::uint64_t ParallelPolicy::partitioning_threshold(mapping::TaskTarget target) const
{
  switch (target) {
    case mapping::TaskTarget::CPU: return cpu_partitioning_threshold_;
    case mapping::TaskTarget::GPU: return gpu_partitioning_threshold_;
    case mapping::TaskTarget::OMP: return omp_partitioning_threshold_;
  }
  LEGATE_ABORT(fmt::format("Invalid target option: {}", detail::to_underlying(target)));
  return 0;
}

bool ParallelPolicy::operator==(const ParallelPolicy& other) const
{
  return streaming_mode() == other.streaming_mode() &&
         overdecompose_factor() == other.overdecompose_factor() &&
         cpu_partitioning_threshold_ == other.cpu_partitioning_threshold_ &&
         gpu_partitioning_threshold_ == other.gpu_partitioning_threshold_ &&
         omp_partitioning_threshold_ == other.omp_partitioning_threshold_;
}

bool ParallelPolicy::operator!=(const ParallelPolicy& other) const { return !(*this == other); }

}  // namespace legate
