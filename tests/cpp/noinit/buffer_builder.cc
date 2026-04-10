/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/buffer_builder.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <utilities/utilities.h>

namespace buffer_builder_test {

constexpr std::uint64_t TEST_PATTERN = 0xDEADBEEFCAFEBABEULL;
constexpr std::uint8_t TEST_BYTE     = 0x42;

using BufferBuilderUnit = DefaultFixture;

TEST_F(BufferBuilderUnit, PackAlreadyAligned)
{
  legate::detail::BufferBuilder builder;
  // The buffer starts empty, so buffer_.data() + 0 is heap-aligned (typically 16-byte).
  // Packing an 8-byte value with 8-byte alignment should succeed without entering the if-block.
  std::uint64_t val = TEST_PATTERN;

  ASSERT_NO_THROW(builder.pack_buffer(&val, sizeof(val), alignof(std::uint64_t)));

  auto buf = builder.to_legion_buffer();

  ASSERT_EQ(buf.get_size(), sizeof(val));
  ASSERT_EQ(std::memcmp(buf.get_ptr(), &val, sizeof(val)), 0);
}

// Cover the TRUE branch of line 53: when the buffer end is misaligned, std::align fails
// and we enter the if-block to resize and re-align.
TEST_F(BufferBuilderUnit, PackMisaligned)
{
  legate::detail::BufferBuilder builder;
  // First, pack a single byte to make the buffer end misaligned for 8-byte alignment.
  std::uint8_t byte_val = TEST_BYTE;

  ASSERT_NO_THROW(builder.pack_buffer(&byte_val, sizeof(byte_val), 1));

  // Now pack an 8-byte value with 8-byte alignment. The buffer end (offset 1) is not
  // 8-byte aligned, so std::align fails and we enter the realignment path.
  std::uint64_t val = TEST_PATTERN;

  ASSERT_NO_THROW(builder.pack_buffer(&val, sizeof(val), alignof(std::uint64_t)));

  auto buf = builder.to_legion_buffer();

  // 1 byte + 7 bytes padding + 8 bytes data = 16
  ASSERT_EQ(buf.get_size(), 16);
  ASSERT_EQ(*static_cast<const std::uint8_t*>(buf.get_ptr()), TEST_BYTE);
  ASSERT_EQ(std::memcmp(static_cast<const std::uint8_t*>(buf.get_ptr()) + 8, &val, sizeof(val)), 0);
}

}  // namespace buffer_builder_test
