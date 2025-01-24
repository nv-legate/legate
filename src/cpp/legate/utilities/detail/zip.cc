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

#include <legate/utilities/detail/zip.h>

#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail::zip_detail {

void throw_unequal_container_sizes()
{
  throw TracedException<std::invalid_argument>{"Arguments to zip_equal() are not all equal"};
}

}  // namespace legate::detail::zip_detail
