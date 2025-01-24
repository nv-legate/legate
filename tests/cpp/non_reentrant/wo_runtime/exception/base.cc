/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/utilities/detail/traced_exception.h>

#include <gtest/gtest.h>

#include <non_reentrant/wo_runtime/exception/common.h>
#include <stdexcept>
#include <string_view>

namespace traced_exception_test {

class TracedExceptionBaseUnit : public TracedExceptionFixture {};

TEST_F(TracedExceptionBaseUnit, Base)
{
  constexpr auto orig_msg = std::string_view{"a very important message"};
  const auto exn          = legate::detail::TracedException<std::runtime_error>{orig_msg.data()};
  const auto& base        = static_cast<const legate::detail::TracedExceptionBase&>(exn);

  ASSERT_EQ(base.raw_what_sv(), orig_msg);
  ASSERT_EQ(base.traced_what(), exn.what());
  ASSERT_EQ(base.traced_what_sv(), exn.what());
}

}  // namespace traced_exception_test
