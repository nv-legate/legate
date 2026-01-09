/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/attachment.h>

#include <legate.h>

#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace attachment_test {

using AttachmentUnit = DefaultFixture;

TEST_F(AttachmentUnit, EmptyDetach)
{
  auto attachment = legate::detail::Attachment{};

  ASSERT_NO_THROW(attachment.detach(false));
}

TEST_F(AttachmentUnit, EmptyMaybeDeallocate)
{
  auto attachment = legate::detail::Attachment{};

  ASSERT_NO_THROW(attachment.maybe_deallocate(false));
}

TEST_F(AttachmentUnit, MaybeDeallocate)
{
  auto logical_region = Legion::LogicalRegion{};
  auto launcher = Legion::IndexAttachLauncher{legion_external_resource_t::LEGION_EXTERNAL_INSTANCE,
                                              logical_region};
  auto external_resources = Legion::Runtime::get_runtime()->attach_external_resources(
    Legion::Runtime::get_context(), launcher);
  std::vector<std::int64_t> buf(1, 0);
  auto size           = buf.size() * sizeof(decltype(buf)::value_type);
  auto realm_resource = std::make_unique<Realm::ExternalMemoryResource>(
    reinterpret_cast<std::uintptr_t>(buf.data()), size, false /* read_only */);
  auto allocation = legate::make_internal_shared<legate::detail::ExternalAllocation>(
    false /* read_only */,
    legate::mapping::StoreTarget::SYSMEM,
    buf.data(),
    size,
    std::move(realm_resource));
  auto allocations =
    std::vector<legate::InternalSharedPtr<legate::detail::ExternalAllocation>>{allocation};
  auto attachment =
    legate::detail::Attachment{std::move(external_resources), std::move(allocations)};

  ASSERT_NO_THROW(attachment.maybe_deallocate(false));
}

}  // namespace attachment_test
