/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
#else
namespace nvtx3 {

// NOLINTBEGIN
class [[maybe_unused]] scoped_range {
 public:
  template <typename... T>
  scoped_range(T&&...) noexcept
  {
  }
};

// NOLINTEND

}  // namespace nvtx3
#endif
