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

#include "legate_defines.h"

#if LegateDefined(LEGATE_USE_CUDA)
#include <nvtx3/nvToolsExt.h>
#else
using nvtxRangeId_t = char;
inline constexpr nvtxRangeId_t nvtxRangeStartA(const char*) noexcept { return 0; }
inline constexpr void nvtxRangeEnd(nvtxRangeId_t) noexcept {}
#endif

namespace legate::nvtx {

class Range {
 public:
  explicit Range(const char* message) noexcept : range_{nvtxRangeStartA(message)} {}

  ~Range() noexcept { nvtxRangeEnd(range_); }

 private:
  nvtxRangeId_t range_;
};

}  // namespace legate::nvtx
