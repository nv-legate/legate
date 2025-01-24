/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/data/detail/scalar.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/deserializer.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace aligned_unpack_test {

using AlignedUnpack = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

class TestDeserializer : public legate::detail::BaseDeserializer<TestDeserializer> {
 public:
  TestDeserializer(const void* args, std::size_t arglen) : BaseDeserializer(args, arglen) {}

  using BaseDeserializer::unpack_impl;
};

TEST_F(AlignedUnpack, Bug1)
{
  legate::Scalar to_pack{complex<double>{123.0, 456.0}};

  legate::detail::BufferBuilder buffer{};
  buffer.pack<bool>(true);
  to_pack.impl()->pack(buffer);

  auto legion_buffer = buffer.to_legion_buffer();
  TestDeserializer dez{legion_buffer.get_ptr(), legion_buffer.get_size()};
  (void)dez.unpack<bool>();
  auto unpacked = legate::Scalar{dez.unpack_scalar()};

  EXPECT_EQ(unpacked.value<complex<double>>(), to_pack.value<complex<double>>());
}

// NOLINTEND(readability-magic-numbers)

}  // namespace aligned_unpack_test
