/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/partitioning/detail/proxy/validate.h>

namespace legate::detail {

// I suppose we could check that the corresponding signature is nonzero for these. So for
// Inputs, and assuming signature->inputs().has_value(), then we would check
// signature->inputs()->contains(1).
//
// But I'm not sure if that's entirely correct.
//
// For example, you could declare a task which must "align all inputs and outputs", but if
// that task ends up taking exactly 0 inputs and outputs, then your alignment constraints
// are *technically* still satisfied.
inline void ValidateVisitor::operator()(const ProxyInputArguments&) const {}

inline void ValidateVisitor::operator()(const ProxyOutputArguments&) const {}

inline void ValidateVisitor::operator()(const ProxyReductionArguments&) const {}

}  // namespace legate::detail
