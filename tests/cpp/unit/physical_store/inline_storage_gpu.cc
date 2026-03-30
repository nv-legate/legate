/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_inline_storage_gpu {

namespace {

class InlineStorageGpuUnit : public DefaultFixture {
 protected:
  void SetUp() override
  {
    DefaultFixture::SetUp();

    auto* const runtime = legate::Runtime::get_runtime();

    if (runtime->get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
      GTEST_SKIP() << "Skipping test due to no GPU available";
    }
  }
};

}  // namespace

TEST_F(InlineStorageGpuUnit, GetPhysicalStoreFBMEM)
{
  constexpr auto DIM   = 2;
  constexpr auto VALUE = std::int32_t{42};
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{DIM, DIM}, legate::int32());

  runtime->issue_fill(legate::LogicalArray{store}, legate::Scalar{VALUE});

  const auto phys = store.get_physical_store(legate::mapping::StoreTarget::FBMEM);

  ASSERT_NE(phys.get_inline_allocation().ptr, nullptr);
}

TEST_F(InlineStorageGpuUnit, GetPhysicalStoreZCMEM)
{
  constexpr auto DIM   = 2;
  constexpr auto VALUE = std::int32_t{77};
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{DIM, DIM}, legate::int32());

  runtime->issue_fill(legate::LogicalArray{store}, legate::Scalar{VALUE});

  const auto phys = store.get_physical_store(legate::mapping::StoreTarget::ZCMEM);

  ASSERT_NE(phys.get_inline_allocation().ptr, nullptr);
}

}  // namespace test_inline_storage_gpu
