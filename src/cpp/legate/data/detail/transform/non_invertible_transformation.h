/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <stdexcept>

namespace legate::detail {

class LEGATE_EXPORT NonInvertibleTransformation : public std::runtime_error {
 public:
  explicit NonInvertibleTransformation();
};

}  // namespace legate::detail
