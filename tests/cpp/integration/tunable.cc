/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace tunable {

using Tunable = DefaultFixture;

namespace {

constexpr std::string_view LIBRARY_NAME = "test_tunable";

[[nodiscard]] const std::vector<legate::Scalar>& TUNABLES()
{
  static const std::vector<legate::Scalar> tunables = {
    legate::Scalar{false},
    legate::Scalar{std::int8_t{12}},
    legate::Scalar{std::int32_t{456}},
    legate::Scalar{std::uint16_t{78}},
    legate::Scalar{std::uint64_t{91011}},
    legate::Scalar{123.0},
    legate::Scalar{legate::Complex<float>{10.0F, 20.0F}},
  };
  return tunables;
}

class LibraryMapper : public legate::mapping::Mapper {
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& /*task*/,
    const std::vector<legate::mapping::StoreTarget>& /*options*/) override
  {
    return {};
  }

  legate::Scalar tunable_value(legate::TunableID tunable_id) override
  {
    return tunable_id < TUNABLES().size() ? TUNABLES().at(tunable_id) : legate::Scalar{};
  }

  std::optional<std::size_t> allocation_pool_size(
    const legate::mapping::Task& /*task*/, legate::mapping::StoreTarget /*memory_kind*/) override
  {
    return std::nullopt;
  }
};

void prepare()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  (void)runtime->create_library(
    LIBRARY_NAME, legate::ResourceConfig{}, std::make_unique<LibraryMapper>());
}

class ScalarEqFn {
 public:
  template <legate::Type::Code CODE>
  bool operator()(const legate::Scalar& lhs, const legate::Scalar& rhs)
  {
    using VAL = legate::type_of_t<CODE>;
    return lhs.value<VAL>() == rhs.value<VAL>();
  }
};

}  // namespace

TEST_F(Tunable, Valid)
{
  prepare();
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(LIBRARY_NAME);

  std::int64_t tunable_id = 0;
  for (const auto& to_compare : TUNABLES()) {
    auto dtype = to_compare.type();
    EXPECT_TRUE(legate::type_dispatch(
      dtype.code(), ScalarEqFn{}, library.get_tunable(tunable_id++, dtype), to_compare));
  }
}

TEST_F(Tunable, Invalid)
{
  prepare();
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(LIBRARY_NAME);

  EXPECT_THROW((void)library.get_tunable(0, legate::string_type()), std::invalid_argument);
  EXPECT_THROW((void)library.get_tunable(0, legate::int64()), std::invalid_argument);
  EXPECT_THROW(
    (void)library.get_tunable(static_cast<std::int64_t>(TUNABLES().size()), legate::bool_()),
    std::invalid_argument);
}

}  // namespace tunable
