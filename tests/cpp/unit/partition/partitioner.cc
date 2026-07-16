/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partitioner.h>

#include <legate.h>

#include <legate/data/detail/transform/transform_stack.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace {

TEST(PartitionerUnit, RejectsLaunchRankMismatch)
{
  const auto launch_domain = legate::Domain{legate::Rect<2>{{0, 0}, {1, 1}}};
  constexpr auto color_shape =
    std::array<std::uint64_t, 3>{std::uint64_t{2}, std::uint64_t{3}, std::uint64_t{4}};
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();

  ASSERT_THAT(
    [&] {
      static_cast<void>(legate::detail::Partitioner::infer_store_projection(
        launch_domain,
        legate::Span<const std::uint64_t>{color_shape.data(), color_shape.size()},
        transform));
    },
    ::testing::ThrowsMessage<std::runtime_error>(
      ::testing::HasSubstr("The launch domain must be 1D")));
}

}  // namespace
