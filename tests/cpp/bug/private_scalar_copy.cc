/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>

#include <utilities/utilities.h>

namespace private_scalar_copy {

class PrivateScalarCopyBug : public DefaultFixture {};

TEST_F(PrivateScalarCopyBug, CopyOwn)
{
  constexpr auto VALUE_1 = std::int32_t{1234};
  constexpr auto VALUE_2 = std::int32_t{4321};

  // Both scalar own their values
  auto scal   = legate::detail::Scalar{VALUE_1};
  auto scal_2 = legate::detail::Scalar{VALUE_2};

  // The pointers should not be the same, means they both allocated their storage on the heap
  // somewhere.
  ASSERT_NE(scal.data(), scal_2.data());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), VALUE_1);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), VALUE_2);

  // This copy should not leak, previously we would clobber the data
  scal = scal_2;

  ASSERT_NE(scal.data(), nullptr);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), VALUE_2);
  ASSERT_NE(scal_2.data(), nullptr);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), VALUE_2);
}

TEST_F(PrivateScalarCopyBug, CopyNotOwn)
{
  constexpr auto VALUE_1 = std::int32_t{1234};
  constexpr auto VALUE_2 = std::int32_t{4321};
  const auto VALUE_1_PTR = std::make_unique<std::int32_t>(VALUE_1);
  const auto VALUE_2_PTR = std::make_unique<std::int32_t>(VALUE_2);

  // Neither scalar own their values
  auto scal   = legate::detail::Scalar{legate::int32().impl(), VALUE_1_PTR.get(), /* copy */ false};
  auto scal_2 = legate::detail::Scalar{legate::int32().impl(), VALUE_2_PTR.get(), /* copy */ false};

  // The pointers should not be the same, they are backed by different unique ptr
  ASSERT_NE(scal.data(), scal_2.data());
  ASSERT_EQ(scal.data(), VALUE_1_PTR.get());
  ASSERT_EQ(scal_2.data(), VALUE_2_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), *VALUE_1_PTR);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), *VALUE_2_PTR);

  // This copy should not leak, previously we would clobber the data
  scal = scal_2;

  ASSERT_EQ(scal.data(), VALUE_2_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), *VALUE_2_PTR);
  ASSERT_EQ(scal_2.data(), VALUE_2_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), *VALUE_2_PTR);
}

TEST_F(PrivateScalarCopyBug, CopySemiOwn)
{
  constexpr auto VALUE_1 = std::int32_t{1234};
  constexpr auto VALUE_2 = std::int32_t{4321};
  const auto VALUE_1_PTR = std::make_unique<std::int32_t>(VALUE_1);

  // The second scalar owns its value, the first does not
  auto scal   = legate::detail::Scalar{legate::int32().impl(), VALUE_1_PTR.get(), /* copy */ false};
  auto scal_2 = legate::detail::Scalar{VALUE_2};

  // The pointers should not be the same, they are backed by a unique ptr and a heap allocation
  ASSERT_NE(scal.data(), scal_2.data());
  ASSERT_EQ(scal.data(), VALUE_1_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), *VALUE_1_PTR);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), VALUE_2);

  // This copy should not leak, previously we would clobber the data
  scal = scal_2;

  ASSERT_NE(scal.data(), VALUE_1_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), VALUE_2);
  ASSERT_NE(scal_2.data(), nullptr);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), VALUE_2);
}

TEST_F(PrivateScalarCopyBug, MoveOwn)
{
  constexpr auto VALUE_1 = std::int32_t{1234};
  constexpr auto VALUE_2 = std::int32_t{4321};

  // Both scalars own their values
  auto scal   = legate::detail::Scalar{VALUE_1};
  auto scal_2 = legate::detail::Scalar{VALUE_2};

  // The pointers should not be the same, means they both allocated their storage on the heap
  // somewhere.
  ASSERT_NE(scal.data(), scal_2.data());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), VALUE_1);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), VALUE_2);

  // This move should not leak, previously we would clobber the data
  scal = std::move(scal_2);

  ASSERT_NE(scal.data(), nullptr);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), VALUE_2);
  ASSERT_EQ(scal_2.data(),  // NOLINT(clang-analyzer-cplusplus.Move, bugprone-use-after-move)
            nullptr);
}

TEST_F(PrivateScalarCopyBug, MoveNotOwn)
{
  constexpr auto VALUE_1 = std::int32_t{1234};
  constexpr auto VALUE_2 = std::int32_t{4321};
  const auto VALUE_1_PTR = std::make_unique<std::int32_t>(VALUE_1);
  const auto VALUE_2_PTR = std::make_unique<std::int32_t>(VALUE_2);

  // Neither scalar owns its values
  auto scal   = legate::detail::Scalar{legate::int32().impl(), VALUE_1_PTR.get(), /* copy */ false};
  auto scal_2 = legate::detail::Scalar{legate::int32().impl(), VALUE_2_PTR.get(), /* copy */ false};

  // The pointers should not be the same, they are backed by different unique ptr
  ASSERT_NE(scal.data(), scal_2.data());
  ASSERT_EQ(scal.data(), VALUE_1_PTR.get());
  ASSERT_EQ(scal_2.data(), VALUE_2_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), *VALUE_1_PTR);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), *VALUE_2_PTR);

  // This move should not leak, previously we would clobber the data
  scal = std::move(scal_2);

  ASSERT_EQ(scal.data(), VALUE_2_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), *VALUE_2_PTR);
  ASSERT_EQ(scal_2.data(),  // NOLINT(clang-analyzer-cplusplus.Move, bugprone-use-after-move)
            nullptr);
}

TEST_F(PrivateScalarCopyBug, MoveSemiOwn)
{
  constexpr auto VALUE_1 = std::int32_t{1234};
  constexpr auto VALUE_2 = std::int32_t{4321};
  const auto VALUE_1_PTR = std::make_unique<std::int32_t>(VALUE_1);

  // The second scalar owns its value, the first does not
  auto scal   = legate::detail::Scalar{legate::int32().impl(), VALUE_1_PTR.get(), /* copy */ false};
  auto scal_2 = legate::detail::Scalar{VALUE_2};

  // The pointers should not be the same, they are backed by a unique ptr and a heap allocation
  ASSERT_NE(scal.data(), scal_2.data());
  ASSERT_EQ(scal.data(), VALUE_1_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), *VALUE_1_PTR);
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal_2.data()), VALUE_2);

  // This move should not leak, previously we would clobber the data
  scal = std::move(scal_2);

  ASSERT_NE(scal.data(), VALUE_1_PTR.get());
  ASSERT_EQ(*static_cast<const std::int32_t*>(scal.data()), VALUE_2);
  ASSERT_EQ(scal_2.data(),  // NOLINT(clang-analyzer-cplusplus.Move, bugprone-use-after-move)
            nullptr);
}

}  // namespace private_scalar_copy
