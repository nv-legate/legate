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

#include <legate_defines.h>

#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

// Include detail/type_info (even though we only use public Type stuff) because that contains
// the formatter for Type::Code
#include <legate/type/detail/type_info.h>
#include <legate/type/type_info.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/dispatch.h>

#include <fmt/format.h>

namespace legate::detail {

void throw_unsupported_dim(std::int32_t dim)
{
  throw TracedException<std::runtime_error>{
    fmt::format("unsupported number of dimensions: {}, must be [1, {}]", dim, LEGATE_MAX_DIM)};
}

void throw_unsupported_type_code(legate::Type::Code code)
{
  throw TracedException<std::runtime_error>{fmt::format("unsupported type code: {}", code)};
}

}  // namespace legate::detail
