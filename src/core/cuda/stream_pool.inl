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

#include "core/cuda/stream_pool.h"

#include <utility>

namespace legate::cuda {

inline StreamView::StreamView(cudaStream_t stream) : valid_{true}, stream_{stream} {}

inline StreamView::operator cudaStream_t() const { return stream_; }

inline StreamView::StreamView(StreamView&& rhs) noexcept
  : valid_{std::exchange(rhs.valid_, false)}, stream_{rhs.stream_}
{
}

inline StreamView& StreamView::operator=(StreamView&& rhs) noexcept
{
  valid_  = std::exchange(rhs.valid_, false);
  stream_ = rhs.stream_;
  return *this;
}

}  // namespace legate::cuda
