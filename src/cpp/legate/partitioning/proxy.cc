/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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
