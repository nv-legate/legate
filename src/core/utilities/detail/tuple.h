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

#include "core/utilities/tuple.h"
#include "core/utilities/typedefs.h"

namespace legate::detail {

// Commonly used conversion routines for tuples

[[nodiscard]] Domain to_domain(const tuple<std::uint64_t>& shape);

[[nodiscard]] DomainPoint to_domain_point(const tuple<std::uint64_t>& shape);

[[nodiscard]] tuple<std::uint64_t> from_domain(const Domain& domain);

}  // namespace legate::detail
