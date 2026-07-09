/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/launcher_arg.h>

#include <legate/operation/detail/store_analyzer.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <gtest/gtest.h>

namespace operation_launcher_arg_test {

using LauncherArgUnit = ::testing::Test;

TEST_F(LauncherArgUnit, WriteOnlyScalarStoreArgNoOpHooks)
{
  auto arg = legate::detail::WriteOnlyScalarStoreArg{/*store=*/nullptr, legate::GlobalRedopID{-1}};
  const auto& base    = static_cast<const legate::detail::AnalyzableBase&>(arg);
  auto unbound_stores = legate::detail::SmallVector<const legate::detail::OutputRegionArg*>{};
  auto analyzer       = legate::detail::StoreAnalyzer{};

  base.record_unbound_stores(unbound_stores);
  base.perform_invalidations();
  arg.analyze(analyzer);

  ASSERT_TRUE(unbound_stores.empty());
}

}  // namespace operation_launcher_arg_test
