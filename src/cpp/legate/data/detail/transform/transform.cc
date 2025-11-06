/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/transform.h>

#include <iostream>

namespace legate::detail {

std::ostream& operator<<(std::ostream& out, const Transform& transform)
{
  transform.print(out);
  return out;
}

}  // namespace legate::detail
