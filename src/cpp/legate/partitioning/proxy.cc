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

#include <legate/partitioning/proxy.h>

#include <legate/partitioning/detail/proxy/constraint.h>
#include <legate/utilities/shared_ptr.h>

#include <utility>

namespace legate {

ProxyConstraint::ProxyConstraint(SharedPtr<detail::ProxyConstraint> impl) : impl_{std::move(impl)}
{
}

ProxyConstraint::~ProxyConstraint() = default;

}  // namespace legate
