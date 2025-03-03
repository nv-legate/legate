/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/zip.h>

#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail::zip_detail {

void throw_unequal_container_sizes()
{
  throw TracedException<std::invalid_argument>{"Arguments to zip_equal() are not all equal"};
}

}  // namespace legate::detail::zip_detail
