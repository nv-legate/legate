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

#pragma once

#include <legate/operation/detail/discard.h>

namespace legate::detail {

inline Discard::Discard(std::uint64_t unique_id,
                        Legion::LogicalRegion region,
                        Legion::FieldID field_id)
  : Operation{unique_id}, region_{std::move(region)}, field_id_{field_id}
{
}

inline Operation::Kind Discard::kind() const { return Kind::DISCARD; }

}  // namespace legate::detail
