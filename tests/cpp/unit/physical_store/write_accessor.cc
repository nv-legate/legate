/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>
#include <numeric>

namespace physical_store_write_accessor_test {

namespace {

constexpr std::uint64_t UINT64_VALUE = 1;

class WriteAccessorFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::PhysicalStore& store) const
  {
    ASSERT_TRUE(store.is_readable());
    ASSERT_TRUE(store.is_writable());
    ASSERT_TRUE(store.is_reducible());

    using T        = legate::type_of_t<CODE>;
    auto op_shape  = store.shape<DIM>();
    auto write_acc = store.write_accessor<T, DIM>();

    if (!op_shape.empty()) {
      for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
        write_acc[*it] = static_cast<T>(2);
        ASSERT_EQ(write_acc[*it], static_cast<T>(2));
      }
    }

    auto bounds = legate::Rect<DIM>{op_shape.lo + legate::Point<DIM>::ONES(), op_shape.hi};

    if (bounds.empty()) {
      return;
    }

    auto write_acc_bounds = store.write_accessor<T, DIM>(bounds);

    for (legate::PointInRectIterator<DIM> it{bounds}; it.valid(); ++it) {
      write_acc_bounds[*it] = static_cast<T>(4);
      ASSERT_EQ(write_acc_bounds[*it], static_cast<T>(4));
    }

    if (LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)) {
      static constexpr auto EXTENTS = 10000;
      auto exceeded_bounds          = legate::Point<DIM>{EXTENTS};

      ASSERT_EXIT(write_acc[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(
        write_acc[(op_shape.hi + legate::Point<DIM>::ONES())], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(write_acc_bounds[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(write_acc_bounds[(bounds.hi + legate::Point<DIM>::ONES())],
                  ::testing::ExitedWithCode(1),
                  "");
    }
  }
};

class WriteAccessorTestTask : public legate::LegateTask<WriteAccessorTestTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

/*static*/ void WriteAccessorTestTask::cpu_variant(legate::TaskContext context)
{
  auto store = context.output(0).data();

  legate::double_dispatch(store.dim(), store.type().code(), WriteAccessorFn{}, store);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_physical_store_write_accessor";

  static void registration_callback(legate::Library library)
  {
    WriteAccessorTestTask::register_variants(library);
  }
};

void test_write_accessor_by_task(legate::LogicalStore& logical_store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, WriteAccessorTestTask::TASK_ID);

  task.add_output(logical_store);
  runtime->submit(std::move(task));
}

void test_write_accessor_future_store(legate::PhysicalStore& store)
{
  ASSERT_TRUE(store.is_readable());
  ASSERT_FALSE(store.is_writable());
  ASSERT_FALSE(store.is_reducible());
  static constexpr auto DIM = 1;

  ASSERT_EQ(store.shape<DIM>().volume(), DIM);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // accessors of beyond the privilege
    ASSERT_THROW(static_cast<void>(store.write_accessor<std::uint64_t, DIM>()),
                 std::invalid_argument);
  }
}

// This test file only performs `ASSERT_EXIT` checks when bounds checking is enabled, so no
// need to turn on the expensive death-test execution mode when that's disabled.
// TODO(mpapadakis): Given the high overhead of death tests (with death_test_style=fast we fork
// on every instance of ASSERT_EXIT, with death_test_style=safe we even restart the whole
// execution from scratch, skipping other checks until we get to this particular ASSERT_EXIT),
// we should only be doing it sparingly, rather than sprinkling it liberally within larger test
// runs.
#if LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)

// NOLINTBEGIN(readability-identifier-naming)

#define PhysicalStoreWriteAccessorUnit PhysicalStoreWriteAccessorDeathTest

// NOLINTEND(readability-identifier-naming)

#endif

class PhysicalStoreWriteAccessorUnit : public RegisterOnceFixture<Config> {};

class BoundStoreWriteAccessorTest
  : public PhysicalStoreWriteAccessorUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, legate::Type>> {};

// NOLINTBEGIN(readability-magic-numbers)

std::vector<std::tuple<legate::Shape, legate::Type>> write_accessor_cases()
{
  std::vector<std::tuple<legate::Shape, legate::Type>> cases = {
    {legate::Shape{10}, legate::uint32()},
    {legate::Shape{20, 10}, legate::bool_()},
    {legate::Shape{5, 2, 4}, legate::float16()},
    {legate::Shape{1, 10, 6, 8}, legate::float32()},
  };

#if LEGATE_MAX_DIM >= 5
  cases.emplace_back(legate::Shape{9, 8, 7, 6, 5}, legate::float64());
#endif
#if LEGATE_MAX_DIM >= 6
  cases.emplace_back(legate::Shape{1, 2, 3, 4, 5, 6}, legate::complex64());
#endif
#if LEGATE_MAX_DIM >= 7
  cases.emplace_back(legate::Shape{1, 1, 1, 1, 1, 1, 1}, legate::complex128());
#endif

  return cases;
}

// NOLINTEND(readability-magic-numbers)

INSTANTIATE_TEST_SUITE_P(PhysicalStoreWriteAccessorUnit,
                         BoundStoreWriteAccessorTest,
                         ::testing::ValuesIn(write_accessor_cases()));

std::vector<std::int32_t> generate_axes(std::uint32_t n)
{
  std::vector<std::int32_t> axes(n);
  std::iota(axes.rbegin(), axes.rend(), 0);
  return axes;
}

}  // namespace

TEST_P(BoundStoreWriteAccessorTest, BoundStore)
{
  auto [shape, type] = GetParam();
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(shape, type);

  test_write_accessor_by_task(logical_store);
}

TEST_P(BoundStoreWriteAccessorTest, TransformedBoundStore)
{
  auto [shape, type] = GetParam();
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(shape, type);
  auto axes          = generate_axes(shape.ndim());

  logical_store = logical_store.transpose(std::move(axes));
  test_write_accessor_by_task(logical_store);
}

TEST_F(PhysicalStoreWriteAccessorUnit, FutureStoreWithTask)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto scalar        = legate::Scalar{UINT64_VALUE};
  auto logical_store = runtime->create_store(legate::Scalar{UINT64_VALUE});

  test_write_accessor_by_task(logical_store);
}

TEST_F(PhysicalStoreWriteAccessorUnit, FutureStore)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_store  = runtime->create_store(legate::Scalar{UINT64_VALUE});
  auto physical_store = logical_store.get_physical_store();

  test_write_accessor_future_store(physical_store);
}

TEST_F(PhysicalStoreWriteAccessorUnit, TransformedFutureStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{UINT64_VALUE});

  logical_store       = logical_store.transpose({0});
  auto physical_store = logical_store.get_physical_store();

  test_write_accessor_future_store(physical_store);
}

TEST_F(PhysicalStoreWriteAccessorUnit, InvalidDim)
{
  auto runtime                       = legate::Runtime::get_runtime();
  static constexpr auto EXTENTS      = 30;
  auto logical_store                 = runtime->create_store({0, EXTENTS}, legate::int16());
  auto store                         = logical_store.get_physical_store();
  constexpr bool VALIDATE_TYPE       = true;
  constexpr std::int32_t INVALID_DIM = 3;

  ASSERT_THROW(static_cast<void>(store.write_accessor<std::int16_t, INVALID_DIM, VALIDATE_TYPE>()),
               std::invalid_argument);
  auto bounds = legate::Rect<INVALID_DIM, std::int16_t>({0, 0, 0}, {0, 0, 0});

  ASSERT_THROW(
    static_cast<void>(store.write_accessor<std::int16_t, INVALID_DIM, VALIDATE_TYPE>(bounds)),
    std::invalid_argument);
}

TEST_F(PhysicalStoreWriteAccessorUnit, InvalidType)
{
  auto runtime                  = legate::Runtime::get_runtime();
  static constexpr auto EXTENTS = 15;
  auto logical_store            = runtime->create_store({0, EXTENTS}, legate::int16());
  auto store                    = logical_store.get_physical_store();
  constexpr bool VALIDATE_TYPE  = true;
  constexpr std::int32_t DIM    = 2;

  ASSERT_THROW(static_cast<void>(store.write_accessor<std::uint64_t, DIM, VALIDATE_TYPE>()),
               std::invalid_argument);
  auto bounds = legate::Rect<DIM, std::uint16_t>{{0, 0}, {0, 0}};

  ASSERT_THROW(static_cast<void>(store.write_accessor<std::int64_t, DIM, VALIDATE_TYPE>(bounds)),
               std::invalid_argument);
}

}  // namespace physical_store_write_accessor_test
