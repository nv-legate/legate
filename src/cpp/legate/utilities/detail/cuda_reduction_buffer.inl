/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/cuda_reduction_buffer.cuh>

namespace legate::detail {

template <typename REDOP>
CUDAReductionBuffer<REDOP>::CUDAReductionBuffer(CUstream stream)
  : buffer_{legate::create_buffer<VAL>(1, Memory::Kind::GPU_FB_MEM, alignof(VAL))},
    ptr_{buffer_.ptr(0)}
{
  static const VAL identity{REDOP::identity};

  Runtime::get_runtime()->get_cuda_driver_api()->mem_cpy_async(
    ptr_, &identity, sizeof(identity), stream);
}

template <typename REDOP>
template <bool EXCLUSIVE>
LEGATE_DEVICE void CUDAReductionBuffer<REDOP>::reduce(const VAL& value) const
{
  REDOP::template fold<EXCLUSIVE /*exclusive*/>(*ptr_, value);
}

template <typename REDOP>
LEGATE_HOST typename CUDAReductionBuffer<REDOP>::VAL CUDAReductionBuffer<REDOP>::read(
  CUstream stream) const
{
  VAL result;
  const auto* driver = Runtime::get_runtime()->get_cuda_driver_api();

  driver->mem_cpy_async(&result, ptr_, sizeof(result), stream);
  driver->stream_synchronize(stream);
  return result;
}

template <typename REDOP>
LEGATE_DEVICE typename CUDAReductionBuffer<REDOP>::VAL CUDAReductionBuffer<REDOP>::read() const
{
  return *ptr_;
}

}  // namespace legate::detail
