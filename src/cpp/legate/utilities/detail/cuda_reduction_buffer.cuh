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

#include <legate/cuda/detail/cuda_driver_types.h>
#include <legate/data/buffer.h>
#include <legate/utilities/macros.h>

namespace legate::detail {

template <typename REDOP>
class CUDAReductionBuffer {
  using VAL = typename REDOP::RHS;

 public:
  using reduction_type = REDOP;

  explicit CUDAReductionBuffer(CUstream stream);

  template <bool EXCLUSIVE>
  LEGATE_DEVICE void reduce(const VAL& value) const;

  [[nodiscard]] LEGATE_HOST VAL read(CUstream stream) const;
  [[nodiscard]] LEGATE_DEVICE VAL read() const;

 private:
  legate::Buffer<VAL> buffer_{};
  VAL* ptr_{};
};

}  // namespace legate::detail

#include <legate/utilities/detail/cuda_reduction_buffer.inl>
