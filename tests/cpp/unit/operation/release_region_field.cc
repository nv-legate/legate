/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/release_region_field.h>

#include <legate.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/scope_guard.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace operation_release_region_field_test {

using ReleaseRegionFieldUnit = DefaultFixture;
using PhysicalState          = legate::detail::LogicalRegionField::PhysicalState;

namespace {

[[nodiscard]] legate::detail::ReleaseRegionField make_release_region_field(
  legate::InternalSharedPtr<PhysicalState> state)
{
  constexpr auto unique_id = std::uint64_t{1};

  return {unique_id, std::move(state), /*unordered=*/false};
}

}  // namespace

TEST_F(ReleaseRegionFieldUnit, SupportsStreamingWithEmptyPhysicalState)
{
  const auto op = make_release_region_field(legate::make_internal_shared<PhysicalState>());

  ASSERT_TRUE(op.supports_streaming());
}

TEST_F(ReleaseRegionFieldUnit, SupportsStreamingRejectsCallbacks)
{
  auto state = legate::make_internal_shared<PhysicalState>();
  state->add_callback([] {});
  const auto op = make_release_region_field(std::move(state));

  ASSERT_FALSE(op.supports_streaming());
}

TEST_F(ReleaseRegionFieldUnit, SupportsStreamingRejectsPhysicalRegion)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Shape{1}, legate::int32());
  const auto& field   = store.impl()->get_region_field();
  auto state          = legate::make_internal_shared<PhysicalState>();

  static_cast<void>(state->ensure_mapping(
    field->region(), field->field_id(), legate::mapping::StoreTarget::SYSMEM));
  // Keep cleanup in a scope guard so a fatal assertion cannot leave the region mapped.
  const auto cleanup = legate::make_scope_guard([&]() noexcept { state->unmap(); });
  const auto op      = make_release_region_field(state);

  ASSERT_FALSE(op.supports_streaming());
}

}  // namespace operation_release_region_field_test
