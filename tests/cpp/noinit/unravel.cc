/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/unravel.h>

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace unravel_test {

using Unravel = ::testing::Test;

TEST_F(Unravel, Unravel1D)
{
  const auto rect    = legate::Rect<1>{legate::Point<1>{0}, legate::Point<1>{99}};
  const auto unravel = legate::detail::Unravel<1>{rect};
  const auto volume  = 100;

  ASSERT_EQ(unravel.volume(), volume);
  ASSERT_FALSE(unravel.empty());

  // Check all unraveled points in 1D
  for (auto i = 0; i < volume; i++) {
    ASSERT_EQ(unravel(i), (legate::Point<1>{i}));
  }
}

TEST_F(Unravel, Unravel2D)
{
  const auto rect    = legate::Rect<2>{legate::Point<2>{-1, -2}, legate::Point<2>{1, 2}};
  const auto unravel = legate::detail::Unravel<2>{rect};

  ASSERT_EQ(unravel.volume(), 15);
  ASSERT_FALSE(unravel.empty());

  // Check some specific points in 2D - start, middle, end
  ASSERT_EQ(unravel(0), (legate::Point<2>{-1, -2}));
  ASSERT_EQ(unravel(8), (legate::Point<2>{0, 1}));
  ASSERT_EQ(unravel(14), (legate::Point<2>{1, 2}));
}

TEST_F(Unravel, Unravel3D)
{
  const auto rect    = legate::Rect<3>{legate::Point<3>{1, 2, 3}, legate::Point<3>{2, 3, 5}};
  const auto unravel = legate::detail::Unravel<3>{rect};

  ASSERT_EQ(unravel.volume(), 12);
  ASSERT_FALSE(unravel.empty());

  // Check some specific points in 3D - start, middle, end
  ASSERT_EQ(unravel(0), (legate::Point<3>{1, 2, 3}));
  ASSERT_EQ(unravel(6), (legate::Point<3>{2, 2, 3}));
  ASSERT_EQ(unravel(11), (legate::Point<3>{2, 3, 5}));
}

TEST_F(Unravel, SingleElement1D)
{
  const auto rect    = legate::Rect<1>{legate::Point<1>{1}, legate::Point<1>{1}};
  const auto unravel = legate::detail::Unravel<1>{rect};

  ASSERT_EQ(unravel.volume(), 1);
  ASSERT_FALSE(unravel.empty());

  ASSERT_EQ(unravel(0), (legate::Point<1>{1}));
}

TEST_F(Unravel, SingleElement2D)
{
  const auto rect    = legate::Rect<2>{legate::Point<2>{2, 3}, legate::Point<2>{2, 3}};
  const auto unravel = legate::detail::Unravel<2>{rect};

  ASSERT_EQ(unravel.volume(), 1);
  ASSERT_FALSE(unravel.empty());

  ASSERT_EQ(unravel(0), (legate::Point<2>{2, 3}));
}

TEST_F(Unravel, SingleElement3D)
{
  const auto rect    = legate::Rect<3>{legate::Point<3>{-1, 2, 3}, legate::Point<3>{-1, 2, 3}};
  const auto unravel = legate::detail::Unravel<3>{rect};

  ASSERT_EQ(unravel.volume(), 1);
  ASSERT_FALSE(unravel.empty());

  ASSERT_EQ(unravel(0), (legate::Point<3>{-1, 2, 3}));
}

TEST_F(Unravel, Empty1D)
{
  const auto rect    = legate::Rect<1>{legate::Point<1>{1}, legate::Point<1>{0}};
  const auto unravel = legate::detail::Unravel<1>{rect};

  ASSERT_EQ(unravel.volume(), 0);
  ASSERT_TRUE(unravel.empty());
}

TEST_F(Unravel, Empty2D)
{
  const auto rect    = legate::Rect<2>{legate::Point<2>{1, 2}, legate::Point<2>{1, 1}};
  const auto unravel = legate::detail::Unravel<2>{rect};

  ASSERT_EQ(unravel.volume(), 0);
  ASSERT_TRUE(unravel.empty());
}

TEST_F(Unravel, Empty3D)
{
  const auto rect    = legate::Rect<3>{legate::Point<3>{1, 2, 3}, legate::Point<3>{0, 1, 2}};
  const auto unravel = legate::detail::Unravel<3>{rect};

  ASSERT_EQ(unravel.volume(), 0);
  ASSERT_TRUE(unravel.empty());
}

}  // namespace unravel_test
