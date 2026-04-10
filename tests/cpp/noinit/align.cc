/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/align.h>

#include <legate/utilities/detail/traced_exception.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utilities/utilities.h>

namespace align_for_unpack_impl_test {

constexpr std::size_t BUF_SIZE = 64;
constexpr std::size_t ALIGN_8  = 8;
constexpr std::size_t ALIGN_16 = 16;
constexpr std::size_t BYTES_8  = 8;

using AlignForUnpackImplUnit = DefaultFixture;

TEST_F(AlignForUnpackImplUnit, ZeroBytes)
{
  alignas(ALIGN_8) std::array<std::byte, BUF_SIZE> buf{};
  void* ptr = buf.data();
  // bytes == 0: always returns {ptr, 0} regardless of capacity or alignment
  const auto [out_ptr, offset] = legate::detail::align_for_unpack_impl(
    ptr, /*capacity=*/BUF_SIZE, /*bytes=*/0, /*align=*/ALIGN_8);

  ASSERT_EQ(out_ptr, buf.data());
  ASSERT_EQ(offset, std::size_t{0});
}

TEST_F(AlignForUnpackImplUnit, SuccessAlreadyAligned)
{
  alignas(ALIGN_8) std::array<std::byte, BUF_SIZE> buf{};
  void* ptr = buf.data();
  // ptr already aligned, capacity >> bytes + align - 1
  // bytes=8, align=8 -> bytes+align-1 = 15 < 64 -> orig_avail_space = 15
  const auto [out_ptr, offset] = legate::detail::align_for_unpack_impl(
    ptr, /*capacity=*/BUF_SIZE, /*bytes=*/BYTES_8, /*align=*/ALIGN_8);

  ASSERT_EQ(out_ptr, buf.data());  // already aligned -> no shift
  ASSERT_EQ(offset, std::size_t{0});
}

TEST_F(AlignForUnpackImplUnit, SuccessMisalignedPtr)
{
  alignas(ALIGN_8) std::array<std::byte, BUF_SIZE> buf{};
  // Shift 1 byte so ptr is not 8-byte aligned
  void* ptr                    = reinterpret_cast<char*>(buf.data()) + 1;
  const auto [out_ptr, offset] = legate::detail::align_for_unpack_impl(
    ptr, /*capacity=*/BUF_SIZE - 1, /*bytes=*/BYTES_8, /*align=*/ALIGN_8);

  // Returned pointer must be 8-byte aligned, offset accounts for the 7-byte alignment gap
  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(out_ptr) % ALIGN_8, std::size_t{0});
  ASSERT_EQ(offset, std::size_t{7});
}

TEST_F(AlignForUnpackImplUnit, ThrowsWhenCapacityTooSmall)
{
  alignas(ALIGN_8) std::array<std::byte, BUF_SIZE> buf{};
  void* ptr = buf.data();

  // Exception path: capacity is too small to fit `bytes` bytes even after alignment.
  // orig_avail_space = min(15, 1) = 1; std::align needs 8 bytes but only 1 available -> fails
  ASSERT_THAT(
    [&] {
      static_cast<void>(legate::detail::align_for_unpack_impl(
        ptr, /*capacity=*/1, /*bytes=*/BYTES_8, /*align=*/ALIGN_8));
    },
    ::testing::ThrowsMessage<legate::detail::TracedException<std::runtime_error>>(
      ::testing::HasSubstr("Failed to align")));
}

TEST_F(AlignForUnpackImplUnit, ThrowsWhenMisalignedAndCapacityTight)
{
  alignas(ALIGN_16) std::array<std::byte, BUF_SIZE> buf{};
  // ptr is 1 byte off from 8-byte alignment -> needs 7-byte shift
  void* ptr = reinterpret_cast<char*>(buf.data()) + 1;

  // misaligned ptr and capacity just barely too tight after the shift.
  // capacity=4: orig_avail_space = min(15, 4) = 4; after 7-byte shift only -3 remain -> fails
  ASSERT_THAT(
    [&] {
      static_cast<void>(legate::detail::align_for_unpack_impl(
        ptr, /*capacity=*/4, /*bytes=*/BYTES_8, /*align=*/ALIGN_8));
    },
    ::testing::ThrowsMessage<legate::detail::TracedException<std::runtime_error>>(
      ::testing::HasSubstr("Failed to align")));
}

}  // namespace align_for_unpack_impl_test
