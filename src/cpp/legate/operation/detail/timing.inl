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

#include <legate/operation/detail/timing.h>

namespace legate::detail {

inline Timing::Timing(std::uint64_t unique_id,
                      Precision precision,
                      InternalSharedPtr<LogicalStore> store)
  : Operation{unique_id}, precision_{precision}, store_{std::move(store)}
{
}

inline Operation::Kind Timing::kind() const { return Kind::TIMING; }

}  // namespace legate::detail
