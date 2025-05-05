/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/utilities/scope_guard.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace remap_double_detach {

using RemapDoubleDetach = DefaultFixture;

// Legate only keeps one inline mapping live at a time. When remapping onto a different memory, we
// need to remove the previous inline mapping. Legate would previously detach in addition to
// unmapping. This would end up prematurely removing the last remaining source of data for the
// store, and surfaced more loudly as a double detach bug on the second unmapping.

TEST_F(RemapDoubleDetach, Bug1)
{
  auto runtime = legate::Runtime::get_runtime();
  if (runtime->get_machine().count(legate::mapping::TaskTarget::OMP) == 0) {
    GTEST_SKIP() << "This test requires NUMA memory";
  }

  auto buffer  = std::array<std::int64_t, 5>{0, 1, 2, 3, 4};
  auto l_store = runtime->create_store(
    legate::Shape{buffer.size()}, legate::int64(), buffer.data(), /*read_only=*/false);
  // NOTE: We must manually detach from the attached memory before we exit the scope, which destroys
  // l_store, since we didn't provide a destruction callback. See the documentation of
  // legate::ExternalAllocation for more details.
  LEGATE_SCOPE_GUARD(l_store.detach());

  {
    auto p_store = l_store.get_physical_store(legate::mapping::StoreTarget::SYSMEM);
    auto acc     = p_store.read_accessor<std::int64_t, 1>();
    ASSERT_EQ(acc[2], 2);
  }
  {
    auto p_store = l_store.get_physical_store(legate::mapping::StoreTarget::SOCKETMEM);
    auto acc     = p_store.read_accessor<std::int64_t, 1>();
    ASSERT_EQ(acc[2], 2);
  }
}

}  // namespace remap_double_detach
