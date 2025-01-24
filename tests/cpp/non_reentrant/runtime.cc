/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/runtime/detail/runtime.h>

#include <legate.h>

#include <legate/data/detail/shape.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_runtime {

using Runtime = DefaultFixture;

TEST_F(Runtime, CreateUnbound)
{
  constexpr std::uint32_t dim = 10;
  auto shape                  = legate::make_internal_shared<legate::detail::Shape>(dim);
  auto runtime                = legate::Runtime::get_runtime();
  ASSERT_THROW(static_cast<void>(runtime->create_array(legate::Shape{shape}, legate::int64())),
               std::invalid_argument);
  ASSERT_THROW(static_cast<void>(runtime->create_store(legate::Shape{shape}, legate::int64())),
               std::invalid_argument);
}

}  // namespace test_runtime
