/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utilities/utilities.h>

namespace logical_store_reinterpret_as_unit {

namespace {

enum class TaskID : std::int8_t { WRITER, READER };

constexpr std::int32_t INT_VAL = 42;

class Writer : public legate::LegateTask<Writer> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{static_cast<std::int32_t>(TaskID::WRITER)}};

  static void cpu_variant(legate::TaskContext ctx)
  {
    auto output = ctx.output(0);

    legate::dim_dispatch(output.dim(), DoWrite{}, &output);
  }

 private:
  class DoWrite {
   public:
    template <std::int32_t DIM>
    void operator()(legate::PhysicalArray* array) const
    {
      const auto shape = array->shape<DIM>();
      const auto acc   = array->data().write_accessor<std::int32_t, DIM>();

      for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
        acc[*it] = INT_VAL;
      }
    }
  };
};

class Reader : public legate::LegateTask<Reader> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{static_cast<std::int32_t>(TaskID::READER)}};

  static void cpu_variant(legate::TaskContext ctx)
  {
    const auto input = ctx.input(0);

    legate::dim_dispatch(input.dim(), DoRead{}, input);
  }

 private:
  class DoRead {
   public:
    template <std::int32_t DIM>
    void operator()(const legate::PhysicalArray& array) const
    {
      const auto shape = array.shape<DIM>();
      const auto acc   = array.data().read_accessor<float, DIM>();
      auto float_val   = float{};

      static_assert(sizeof(float_val) == sizeof(INT_VAL));
      // Need to memcpy here in order to do a "true" bitcast. A reinterpret_cast() may or may
      // not result in the compilers actually generating the conversion, since type-punning
      // with reinterpret_cast is UB.
      std::memcpy(&float_val, &INT_VAL, sizeof(float_val));
      for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
        ASSERT_EQ(acc[*it], float_val);
      }
    }
  };
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_logical_store_reinterpret_as";

  static void registration_callback(legate::Library library)
  {
    Writer::register_variants(library);
    Reader::register_variants(library);
  }
};

class LogicalStoreUnit : public RegisterOnceFixture<Config> {};

class ReinterpretAs : public LogicalStoreUnit,
                      public ::testing::WithParamInterface<legate::Shape> {};

static_assert(LEGATE_MAX_DIM >= 3, "This test requires at least DIM 3");

INSTANTIATE_TEST_SUITE_P(LogicalStoreUnit,
                         ReinterpretAs,
                         ::testing::Values(legate::Shape{3},
                                           legate::Shape{3, 2},
                                           legate::Shape{2, 3, 2}));

}  // namespace

TEST_P(ReinterpretAs, Basic)
{
  auto&& shape  = GetParam();
  auto* runtime = legate::Runtime::get_runtime();
  auto store    = runtime->create_store(shape, legate::int32());
  auto library  = runtime->find_library(Config::LIBRARY_NAME);

  {
    auto task = runtime->create_task(library, Writer::TASK_CONFIG.task_id());

    task.add_output(store);
    runtime->submit(std::move(task));
  }

  {
    auto task = runtime->create_task(library, Reader::TASK_CONFIG.task_id());

    task.add_input(store.reinterpret_as(legate::float32()));
    runtime->submit(std::move(task));
  }
}

TEST_F(ReinterpretAs, Example)
{
  const auto shape = legate::Shape{4};
  auto* runtime    = legate::Runtime::get_runtime();

  /// [Reinterpret store data]
  // Create a store of some shape filled with int32 data.
  constexpr std::int32_t minus_one = -1;
  const auto store                 = runtime->create_store(shape, legate::int32());

  runtime->issue_fill(store, legate::Scalar{minus_one});
  // Reinterpret the underlying data as unsigned 32-bit integers.
  auto reinterp_store = store.reinterpret_as(legate::uint32());
  // Our new store should have the same type as it was reinterpreted to.
  ASSERT_EQ(reinterp_store.type(), legate::uint32());
  // Our old store still has the same type though.
  ASSERT_EQ(store.type(), legate::int32());
  // Both stores should refer to the same underlying storage.
  ASSERT_TRUE(store.equal_storage(reinterp_store));

  const auto phys_store = reinterp_store.get_physical_store();
  const auto acc        = phys_store.read_accessor<std::uint32_t, 1>();

  std::uint32_t interp_value;
  // Need to memcpy here in order to do a "true" bitcast. A reinterpret_cast() may or may not
  // result in the compilers generating the conversion, since type-punning with
  // reinterpret_cast is UB.
  std::memcpy(&interp_value, &minus_one, sizeof(minus_one));
  for (auto it = legate::PointInRectIterator<1>{phys_store.shape<1>()}; it.valid(); ++it) {
    ASSERT_EQ(acc[*it], interp_value);
  }
  /// [Reinterpret store data]
}

TEST_F(ReinterpretAs, BadSize)
{
  auto* runtime    = legate::Runtime::get_runtime();
  const auto store = runtime->create_store(legate::Shape{1}, legate::int32());

  ASSERT_THAT([&] { static_cast<void>(store.reinterpret_as(legate::int64())); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("size of the types must be equal")));
}

TEST_F(ReinterpretAs, BadAlignment)
{
  auto* runtime = legate::Runtime::get_runtime();
  // int64 has size=8, alignment=8
  const auto store = runtime->create_store(legate::Shape{1}, legate::int64());
  // struct(int32, int32) with aligned=true has size=8, alignment=4
  const auto struct_type = legate::struct_type(/*align=*/true, legate::int32(), legate::int32());

  // Same size but different alignment should throw
  ASSERT_THAT([&] { static_cast<void>(store.reinterpret_as(struct_type)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("alignment of the types must be equal")));
}

}  // namespace logical_store_reinterpret_as_unit
