/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/deserializer.h>

#include <cuda/std/complex>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace aligned_unpack_test {

using AlignedUnpack = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

class TestDeserializer : public legate::detail::BaseDeserializer<TestDeserializer> {
 public:
  TestDeserializer(const void* args, std::size_t arglen) : BaseDeserializer{args, arglen} {}

  using BaseDeserializer::unpack_impl;
};

TEST_F(AlignedUnpack, Bug1)
{
  const legate::Scalar to_pack{legate::Complex<double>{123.0, 456.0}};

  legate::detail::BufferBuilder buffer{};
  buffer.pack<bool>(true);
  to_pack.impl()->pack(buffer);

  auto legion_buffer = buffer.to_legion_buffer();
  TestDeserializer dez{legion_buffer.get_ptr(), legion_buffer.get_size()};
  static_cast<void>(dez.unpack<bool>());
  auto unpacked = legate::Scalar{dez.unpack_scalar()};

  EXPECT_EQ(unpacked.value<legate::Complex<double>>(), to_pack.value<legate::Complex<double>>());
}

// NOLINTEND(readability-magic-numbers)

}  // namespace aligned_unpack_test
