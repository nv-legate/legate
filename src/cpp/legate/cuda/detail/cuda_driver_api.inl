/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/cuda/detail/cuda_driver_api.h>

#include <memory>

namespace legate::cuda::detail {

template <typename... T>
void CUDADriverAPI::launch_kernel(CUfunction f,
                                  Dim3 grid_dim,
                                  Dim3 block_dim,
                                  std::size_t shared_mem_bytes,
                                  CUstream stream,
                                  T&&... args) const
{
  void* kernel_params[] = {static_cast<void*>(std::addressof(args))...};

  launch_kernel_direct(f, grid_dim, block_dim, shared_mem_bytes, stream, kernel_params, nullptr);
}

template <typename... T>
void CUDADriverAPI::launch_kernel(CUkernel f,
                                  Dim3 grid_dim,
                                  Dim3 block_dim,
                                  std::size_t shared_mem_bytes,
                                  CUstream stream,
                                  T&&... args) const
{
  launch_kernel(reinterpret_cast<CUfunction>(f),
                grid_dim,
                block_dim,
                shared_mem_bytes,
                stream,
                std::forward<T>(args)...);
}

}  // namespace legate::cuda::detail
