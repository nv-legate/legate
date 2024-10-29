/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate/cuda/detail/cuda_driver_api.h"

namespace legate::cuda::detail {

template <typename T, typename U>
CUresult CUDADriverAPI::mem_cpy_async(T* dst,
                                      const U* src,
                                      std::size_t num_bytes,
                                      CUstream stream) const
{
  return mem_cpy_async(reinterpret_cast<std::uintptr_t>(dst),
                       reinterpret_cast<std::uintptr_t>(src),
                       num_bytes,
                       stream);
}

template <typename T, typename U>
CUresult CUDADriverAPI::mem_cpy(T* dst, const U* src, std::size_t num_bytes) const
{
  return mem_cpy(
    reinterpret_cast<std::uintptr_t>(dst), reinterpret_cast<std::uintptr_t>(src), num_bytes);
}

}  // namespace legate::cuda::detail
