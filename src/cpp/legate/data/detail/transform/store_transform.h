/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/transform.h>
#include <legate/utilities/detail/small_vector.h>

#include <cstdint>

namespace legate::detail {

class StoreTransform : public Transform {
 public:
  [[nodiscard]] virtual std::int32_t target_ndim(std::int32_t source_ndim) const     = 0;
  virtual void find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const = 0;
};

}  // namespace legate::detail
