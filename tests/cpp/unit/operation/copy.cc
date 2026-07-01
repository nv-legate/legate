/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/copy.h>

#include <legate.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/copy_launcher.h>
#include <legate/operation/detail/operation.h>
#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utilities/utilities.h>
#include <utility>

namespace operation_copy_test {

using CopyUnit = DefaultFixture;

TEST_F(CopyUnit, Kind)
{
  constexpr auto unique_id = std::uint64_t{1};
  constexpr auto priority  = std::int32_t{0};

  auto* const runtime = legate::Runtime::get_runtime();
  auto target         = runtime->create_store(legate::Shape{1}, legate::int32());
  auto source         = runtime->create_store(legate::Shape{1}, legate::int32());
  const auto copy     = legate::detail::Copy{target.impl(),
                                             source.impl(),
                                             unique_id,
                                             priority,
                                             runtime->impl()->get_machine(),
                                             std::nullopt};

  ASSERT_EQ(copy.kind(), legate::detail::Operation::Kind::COPY);
}

TEST_F(CopyUnit, ToStringIncludesProvenance)
{
  constexpr auto unique_id = std::uint64_t{1};
  constexpr auto priority  = std::int32_t{0};
  const auto provenance    = std::string{"operation-to-string-provenance"};
  const auto expected_base = std::string{"Copy:"} + std::to_string(unique_id);

  auto* const runtime = legate::Runtime::get_runtime();
  auto target         = runtime->create_store(legate::Shape{1}, legate::int32());
  auto source         = runtime->create_store(legate::Shape{1}, legate::int32());
  auto expected       = expected_base;
  const auto copy     = [&] {
    const auto scope = legate::Scope{provenance};

    return legate::detail::Copy{target.impl(),
                                source.impl(),
                                unique_id,
                                priority,
                                runtime->impl()->get_machine(),
                                std::nullopt};
  }();

  expected.append("[").append(provenance).append("]");
  ASSERT_EQ(copy.to_string(/*show_provenance=*/true), expected);
  ASSERT_EQ(copy.to_string(/*show_provenance=*/false), expected_base);
}

TEST_F(CopyUnit, CopyArgCreatesRequirement)
{
  constexpr auto req_idx   = std::uint32_t{0};
  constexpr auto privilege = LEGION_READ_ONLY;

  auto* const runtime      = legate::Runtime::get_runtime();
  auto store               = runtime->create_store(legate::Shape{1}, legate::int32());
  const auto& store_impl   = store.impl();
  const auto& region_field = store_impl->get_region_field();
  const auto field_id      = region_field->field_id();
  auto copy_arg            = legate::detail::CopyArg{
    req_idx, store_impl.get(), field_id, privilege, legate::detail::StoreProjection{}};
  auto moved_copy_arg    = std::move(copy_arg);
  const auto requirement = moved_copy_arg.create_requirement<true>();

  ASSERT_EQ(requirement.region, region_field->region());
  ASSERT_EQ(requirement.privilege, privilege);
  ASSERT_EQ(requirement.privilege_fields.count(field_id), 1);
}

}  // namespace operation_copy_test
