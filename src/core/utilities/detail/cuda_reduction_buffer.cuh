/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/cuda/cuda.h"
#include "core/data/buffer.h"

namespace legate::detail {

template <typename REDOP>
class CUDAReductionBuffer {
 private:
  using VAL = typename REDOP::RHS;

 public:
  explicit CUDAReductionBuffer(cudaStream_t stream)
    : buffer_{legate::create_buffer<VAL>(1, Memory::Kind::GPU_FB_MEM)}
  {
    VAL identity{REDOP::identity};
    ptr_ = buffer_.ptr(0);
    LegateCheckCUDA(
      cudaMemcpyAsync(ptr_, &identity, sizeof(identity), cudaMemcpyHostToDevice, stream));
  }

  template <bool EXCLUSIVE>
  __device__ void reduce(const VAL& value) const
  {
    REDOP::template fold<EXCLUSIVE /*exclusive*/>(*ptr_, value);
  }

  [[nodiscard]] __host__ VAL read(cudaStream_t stream) const
  {
    VAL result{REDOP::identity};
    LegateCheckCUDA(cudaMemcpyAsync(&result, ptr_, sizeof(result), cudaMemcpyDeviceToHost, stream));
    LegateCheckCUDA(cudaStreamSynchronize(stream));
    return result;
  }

  [[nodiscard]] __device__ VAL read() const { return *ptr_; }

 private:
  legate::Buffer<VAL> buffer_{};
  VAL* ptr_{};
};

}  // namespace legate::detail
