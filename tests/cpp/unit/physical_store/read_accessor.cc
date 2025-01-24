/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <numeric>
#include <utilities/utilities.h>

namespace physical_store_read_accessor_test {

namespace {

constexpr float FLOAT_VALUE = 99.0F;

class ReadAccessorFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(legate::TaskContext context, const legate::PhysicalStore& store) const
  {
    using T       = legate::type_of_t<CODE>;
    auto value    = context.scalar(0).value<T>();
    auto read_acc = store.read_accessor<T, DIM>();
    auto op_shape = store.shape<DIM>();

    if (!op_shape.empty()) {
      for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
        ASSERT_EQ(read_acc[*it], static_cast<T>(value));
      }
    }

    auto bounds = legate::Rect<DIM>{op_shape.lo + legate::Point<DIM>::ONES(), op_shape.hi};

    if (bounds.empty()) {
      return;
    }

    for (legate::PointInRectIterator<DIM> it{bounds}; it.valid(); ++it) {
      ASSERT_EQ(read_acc[*it], static_cast<T>(value));
    }

    if (LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)) {
      // access store with exceeded bounds
      auto read_acc_bounds          = store.read_accessor<T, DIM>(bounds);
      static constexpr auto EXTENTS = 10000;
      auto exceeded_bounds          = legate::Point<DIM>{EXTENTS};

      ASSERT_EXIT(read_acc[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(
        read_acc[(op_shape.hi + legate::Point<DIM>{100})], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(read_acc_bounds[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(read_acc_bounds[(bounds.hi + legate::Point<DIM>::ONES())],
                  ::testing::ExitedWithCode(1),
                  "");
    }

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      // accessors of beyond the privilege
      ASSERT_THROW(static_cast<void>(store.write_accessor<T, DIM>()), std::invalid_argument);
      ASSERT_THROW(static_cast<void>(store.read_write_accessor<T, DIM>()), std::invalid_argument);
      ASSERT_THROW(static_cast<void>(store.reduce_accessor<legate::SumReduction<T>, false, DIM>()),
                   std::invalid_argument);
    }
  }
};

class ReadAccessorTestTask : public legate::LegateTask<ReadAccessorTestTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

/*static*/ void ReadAccessorTestTask::cpu_variant(legate::TaskContext context)
{
  auto store = context.input(0).data();

  legate::double_dispatch(store.dim(), store.type().code(), ReadAccessorFn{}, context, store);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_physical_store_read_accessor";

  static void registration_callback(legate::Library library)
  {
    ReadAccessorTestTask::register_variants(library);
  }
};

void test_read_accessor_by_task(legate::LogicalStore& logical_store, legate::Scalar& scalar)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, ReadAccessorTestTask::TASK_ID);

  runtime->issue_fill(logical_store, scalar);
  task.add_input(logical_store);
  task.add_scalar_arg(scalar);
  runtime->submit(std::move(task));
}

void test_read_accessor_future_store(legate::PhysicalStore& store)
{
  ASSERT_TRUE(store.is_readable());
  ASSERT_FALSE(store.is_writable());
  ASSERT_FALSE(store.is_reducible());
  static constexpr auto DIM = 1;

  ASSERT_EQ(store.shape<DIM>().volume(), DIM);
  auto read_acc = store.read_accessor<float, DIM>();

  ASSERT_EQ(read_acc[0], store.scalar<float>());

  if (LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)) {
    // access store with exceeded bounds
    static constexpr auto EXTENTS = 1000;
    auto exceeded_bounds          = legate::Point<DIM>{EXTENTS};

    ASSERT_EXIT(read_acc[exceeded_bounds], ::testing::ExitedWithCode(1), "");
  }
}

std::vector<std::int32_t> generate_axes(std::uint32_t n)
{
  std::vector<std::int32_t> axes(n);
  std::iota(axes.rbegin(), axes.rend(), 0);
  return axes;
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

#define PhysicalStoreReadAccessorUnit PhysicalStoreReadAccessorDeathTest

// NOLINTEND(readability-identifier-naming)

#endif

class PhysicalStoreReadAccessorUnit : public RegisterOnceFixture<Config> {};

class BoundStoreReadAccessorTest
  : public PhysicalStoreReadAccessorUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, legate::Type, legate::Scalar>> {
};

// NOLINTBEGIN(readability-magic-numbers)

std::vector<std::tuple<legate::Shape, legate::Type, legate::Scalar>> read_accessor_cases()
{
  std::vector<std::tuple<legate::Shape, legate::Type, legate::Scalar>> cases = {
    {legate::Shape{1}, legate::uint32(), legate::Scalar{std::uint32_t{100}, legate::uint32()}},
    {legate::Shape{3, 3}, legate::bool_(), legate::Scalar{true}},
    {legate::Shape{2, 1, 5}, legate::float16(), legate::Scalar{static_cast<__half>(FLOAT_VALUE)}},
    {legate::Shape{3, 4, 1, 2}, legate::float32(), legate::Scalar{FLOAT_VALUE}},
  };

#if LEGATE_MAX_DIM >= 5
  cases.emplace_back(
    legate::Shape{1, 2, 5, 7, 5}, legate::uint64(), legate::Scalar{std::uint64_t{90}});
#endif
#if LEGATE_MAX_DIM >= 6
  cases.emplace_back(
    legate::Shape{1, 2, 3, 4, 5, 6}, legate::int32(), legate::Scalar{std::int32_t{20}});
#endif
#if LEGATE_MAX_DIM >= 7
  cases.emplace_back(legate::Shape{1, 1, 1, 1, 1, 1, 1},
                     legate::complex128(),
                     legate::Scalar{complex<double>{FLOAT_VALUE, FLOAT_VALUE}});
#endif

  return cases;
}

// NOLINTEND(readability-magic-numbers)

INSTANTIATE_TEST_SUITE_P(PhysicalStoreReadAccessorUnit,
                         BoundStoreReadAccessorTest,
                         ::testing::ValuesIn(read_accessor_cases()));

}  // namespace

TEST_P(BoundStoreReadAccessorTest, BoundStore)
{
  auto [shape, type, scalar] = GetParam();
  auto runtime               = legate::Runtime::get_runtime();
  auto logical_store         = runtime->create_store(shape, type);

  test_read_accessor_by_task(logical_store, scalar);
}

TEST_P(BoundStoreReadAccessorTest, TransformedBoundStore)
{
  auto [shape, type, scalar] = GetParam();
  auto runtime               = legate::Runtime::get_runtime();
  auto logical_store         = runtime->create_store(shape, type);
  auto axes                  = generate_axes(shape.ndim());

  logical_store = logical_store.transpose(std::move(axes));
  test_read_accessor_by_task(logical_store, scalar);
}

TEST_F(PhysicalStoreReadAccessorUnit, FutureStoreWithTask)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto scalar        = legate::Scalar{FLOAT_VALUE};
  auto logical_store = runtime->create_store(legate::Scalar{FLOAT_VALUE});

  test_read_accessor_by_task(logical_store, scalar);
}

TEST_F(PhysicalStoreReadAccessorUnit, FutureStore)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_store  = runtime->create_store(legate::Scalar{FLOAT_VALUE});
  auto physical_store = logical_store.get_physical_store();

  test_read_accessor_future_store(physical_store);
}

TEST_F(PhysicalStoreReadAccessorUnit, TransformedFutureStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{FLOAT_VALUE});

  logical_store       = logical_store.transpose({0});
  auto physical_store = logical_store.get_physical_store();

  test_read_accessor_future_store(physical_store);
}

TEST_F(PhysicalStoreReadAccessorUnit, InvalidDim)
{
  auto runtime                  = legate::Runtime::get_runtime();
  static constexpr auto EXTENTS = 10;
  auto logical_store            = runtime->create_store(legate::Shape{1, EXTENTS}, legate::int16());
  auto store                    = logical_store.get_physical_store();
  constexpr bool VALIDATE_TYPE  = true;
  constexpr std::int32_t INVALID_DIM = 3;

  ASSERT_THROW(static_cast<void>(store.read_accessor<std::int16_t, INVALID_DIM, VALIDATE_TYPE>()),
               std::invalid_argument);
  auto bounds = legate::Rect<INVALID_DIM, std::int16_t>({0, 0, 0}, {0, 0, 0});

  ASSERT_THROW(
    static_cast<void>(store.read_accessor<std::int16_t, INVALID_DIM, VALIDATE_TYPE>(bounds)),
    std::invalid_argument);
}

TEST_F(PhysicalStoreReadAccessorUnit, InvalidType)
{
  auto runtime                  = legate::Runtime::get_runtime();
  static constexpr auto EXTENTS = 20;
  auto logical_store            = runtime->create_store({0, EXTENTS}, legate::int16());
  auto store                    = logical_store.get_physical_store();
  constexpr bool VALIDATE_TYPE  = true;
  constexpr std::int32_t DIM    = 2;

  ASSERT_THROW(static_cast<void>(store.read_accessor<std::int32_t, DIM, VALIDATE_TYPE>()),
               std::invalid_argument);
  auto bounds = legate::Rect<DIM, std::uint16_t>{{0, 0}, {0, 0}};

  ASSERT_THROW(static_cast<void>(store.read_accessor<std::uint32_t, DIM, VALIDATE_TYPE>(bounds)),
               std::invalid_argument);
}

}  // namespace physical_store_read_accessor_test
