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

#include "config.hpp"
#include "legate.h"

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
constexpr ResourceConfig LEGATE_STL_RESOURCE_CONFIG = {
  1024, /* max_tasks{1024}; */
  1024, /* max_dyn_tasks{0}; */
  64,   /* max_reduction_ops{}; */
  0,    /* max_projections{}; */
  0     /* max_shardings{}; */
};

////////////////////////////////////////////////////////////////////////////////////////////////////
class initialize_library {
 public:
  initialize_library(int argc, char* argv[]) : result_{legate::start(argc, argv)}
  {
    if (result() == 0) {
      library_ =
        legate::Runtime::get_runtime()->create_library("legate.stl", LEGATE_STL_RESOURCE_CONFIG);
    }
  }

  ~initialize_library()
  {
    if (result() == 0) {
      result_ = legate::finish();
    }
  }

  [[nodiscard]] std::int32_t result() const { return result_; }

 private:
  std::int32_t result_{};
  Library library_{};
};

}  // namespace legate::stl

#include "suffix.hpp"
