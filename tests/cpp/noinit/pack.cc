/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/pack.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utilities/utilities.h>

namespace pack_buffer_test {

namespace {

constexpr std::size_t BUF_SIZE = 64;

using PackBufferUnit = DefaultFixture;

}  // namespace

TEST_F(PackBufferUnit, PackBufferZeroElements)
{
  char buf[BUF_SIZE]       = {};
  void* ptr                = buf;
  const std::size_t cap    = sizeof(buf);
  const std::int32_t dummy = 42;

  const auto [out_ptr, out_cap] = legate::detail::pack_buffer(ptr, cap, /*nelem=*/0, &dummy);

  ASSERT_EQ(out_ptr, ptr);
  ASSERT_EQ(out_cap, cap);
}

TEST_F(PackBufferUnit, UnpackBufferZeroElements)
{
  const char buf[BUF_SIZE] = {};
  const void* ptr          = buf;
  const std::size_t cap    = sizeof(buf);
  std::int32_t dummy       = 0;
  std::int32_t* value_ptr  = &dummy;

  const auto [out_ptr, out_cap] = legate::detail::unpack_buffer(ptr, cap, /*nelem=*/0, &value_ptr);

  ASSERT_EQ(out_ptr, ptr);
  ASSERT_EQ(out_cap, cap);
}

TEST_F(PackBufferUnit, PackUnpackRoundTripNonZeroElements)
{
  alignas(std::int32_t) char buf[BUF_SIZE] = {};
  constexpr std::size_t nelem              = 4;
  const std::int32_t src[nelem]            = {1, -2, 3, -4};
  const std::size_t cap                    = sizeof(buf);
  const std::size_t data_size              = nelem * sizeof(std::int32_t);

  const auto [pack_end, pack_cap] =
    legate::detail::pack_buffer(static_cast<void*>(buf), cap, nelem, src);

  ASSERT_EQ(static_cast<char*>(pack_end), buf + data_size);
  ASSERT_EQ(pack_cap, cap - data_size);
  ASSERT_EQ(std::memcmp(buf, src, data_size), 0);

  std::int32_t dst[nelem] = {};
  std::int32_t* dst_ptr   = dst;
  const auto [unpack_end, unpack_cap] =
    legate::detail::unpack_buffer(static_cast<const void*>(buf), cap, nelem, &dst_ptr);

  ASSERT_EQ(static_cast<const char*>(unpack_end), buf + data_size);
  ASSERT_EQ(unpack_cap, cap - data_size);
  for (std::size_t i = 0; i < nelem; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
}

}  // namespace pack_buffer_test
