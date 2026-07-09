/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/launch_domain_resolver.h>

#include <legate.h>

#include <gtest/gtest.h>

namespace {

TEST(LaunchDomainResolver, ConflictingUnboundDims)
{
  legate::detail::LaunchDomainResolver resolver;

  resolver.record_launch_domain(legate::Domain{legate::Rect<2>{{0, 0}, {1, 1}}});
  resolver.record_unbound_store(1);
  resolver.record_unbound_store(2);

  ASSERT_FALSE(resolver.resolve_launch_domain().is_valid());
}

TEST(LaunchDomainResolver, OneDimensionalFallbackRejectsHigherDim)
{
  legate::detail::LaunchDomainResolver resolver;

  resolver.record_launch_domain(legate::Domain{legate::Rect<1>{0, 3}});
  resolver.record_launch_domain(legate::Domain{legate::Rect<2>{{0, 0}, {1, 1}}});
  resolver.record_unbound_store(2);

  ASSERT_FALSE(resolver.resolve_launch_domain().is_valid());
}

}  // namespace
