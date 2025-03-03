/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/cuda/stream_pool.h>

#include <utility>

namespace legate::cuda {

inline StreamView::StreamView(CUstream stream) : valid_{true}, stream_{stream} {}

inline StreamView::operator CUstream() const { return stream_; }

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
