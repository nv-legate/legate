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

#ifdef LEGATE_USE_CUDA

#include <nvtx3/nvToolsExt.h>

namespace legate::nvtx {

class Range {
 public:
  Range(const char* message) { range_ = nvtxRangeStartA(message); }
  ~Range() { nvtxRangeEnd(range_); }

 private:
  nvtxRangeId_t range_;
};

}  // namespace legate::nvtx

#endif
