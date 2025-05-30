/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/utilities/mdspan.h>

#include <fmt/ranges.h>

#include <gtest/gtest.h>

#include <numeric>
#include <utilities/utilities.h>

namespace physical_store_read_write_accessor_test {

namespace {

constexpr std::uint64_t UINT64_VALUE = 1;
constexpr float FLOAT_VALUE          = 11.0F;

class ReadWriteAccessorFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(legate::PhysicalStore& store) const
  {
    using T = legate::type_of_t<CODE>;

    ASSERT_TRUE(store.is_readable());
    ASSERT_TRUE(store.is_writable());
    ASSERT_TRUE(store.is_reducible());

    const auto LEGION_ACC_INIT_VAL        = static_cast<T>(32);
    const auto LEGION_ACC_BOUNDS_INIT_VAL = static_cast<T>(66);
    const auto SPAN_ACC_INIT_VAL          = static_cast<T>(5);
    const auto SPAN_ACC_BOUNDS_INIT_VAL   = static_cast<T>(10);

    // Don't check this stuff for bools. The point here is that none of the above values are
    // equal to each other, but that requires there to be 4 unique values for the type (which
    // bool does not have).
    if constexpr (!std::is_same_v<T, bool>) {
      ASSERT_NE(LEGION_ACC_INIT_VAL, SPAN_ACC_INIT_VAL);
      ASSERT_NE(LEGION_ACC_INIT_VAL, LEGION_ACC_BOUNDS_INIT_VAL);
      ASSERT_NE(LEGION_ACC_BOUNDS_INIT_VAL, SPAN_ACC_BOUNDS_INIT_VAL);
    }

    const auto op_shape       = store.shape<DIM>();
    const auto op_shape_empty = op_shape.empty();
    std::size_t legion_acc_it = 0;

    if (!op_shape_empty) {
      const auto legion_acc = store.read_write_accessor<T, DIM>();

      for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
        legion_acc[*it] = LEGION_ACC_INIT_VAL;
        ASSERT_EQ(legion_acc[*it], LEGION_ACC_INIT_VAL);
        ++legion_acc_it;
      }
    }

    {
      const auto span_acc     = store.span_read_write_accessor<T, DIM>();
      std::size_t span_acc_it = 0;

      ASSERT_EQ(span_acc.size(), legion_acc_it);
      static_assert(span_acc.rank() == DIM);
      for (auto& v : legate::flatten(span_acc)) {
        if (op_shape_empty) {
          GTEST_FAIL() << "Should not iterate empty shape for unbounded span accessor";
        }
        ASSERT_EQ(v, LEGION_ACC_INIT_VAL);
        v = SPAN_ACC_INIT_VAL;
        ASSERT_EQ(v, SPAN_ACC_INIT_VAL);
        ++span_acc_it;
      }
      ASSERT_EQ(legion_acc_it, span_acc_it);
      legion_acc_it = 0;
    }

    if (!op_shape_empty) {
      const auto legion_acc = store.read_write_accessor<T, DIM>();

      for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
        ASSERT_EQ(legion_acc[*it], SPAN_ACC_INIT_VAL);
      }
    }

    const auto bounds = legate::Rect<DIM>{op_shape.lo + legate::Point<DIM>::ONES(), op_shape.hi};
    const auto bounds_empty = bounds.empty();

    if (!bounds_empty) {
      auto legion_acc = store.read_write_accessor<T, DIM>(bounds);

      for (legate::PointInRectIterator<DIM> it{bounds}; it.valid(); ++it) {
        ASSERT_EQ(legion_acc[*it], SPAN_ACC_INIT_VAL);
        legion_acc[*it] = LEGION_ACC_BOUNDS_INIT_VAL;
        ASSERT_EQ(legion_acc[*it], LEGION_ACC_BOUNDS_INIT_VAL);
        ++legion_acc_it;
      }
    }

    {
      const auto span_acc     = store.span_read_write_accessor<T, DIM>(bounds);
      std::size_t span_acc_it = 0;

      ASSERT_EQ(span_acc.size(), legion_acc_it);
      static_assert(span_acc.rank() == DIM);
      for (auto& v : legate::flatten(span_acc)) {
        if (bounds_empty) {
          GTEST_FAIL() << "Should not iterate empty bounds for bounded span accessor";
        }
        ASSERT_EQ(v, LEGION_ACC_BOUNDS_INIT_VAL);
        v = SPAN_ACC_BOUNDS_INIT_VAL;
        ASSERT_EQ(v, SPAN_ACC_BOUNDS_INIT_VAL);
        ++span_acc_it;
      }
      ASSERT_EQ(legion_acc_it, span_acc_it);
      legion_acc_it = 0;
    }

    if (!bounds_empty) {
      const auto legion_acc = store.read_write_accessor<T, DIM>(bounds);

      for (legate::PointInRectIterator<DIM> it{bounds}; it.valid(); ++it) {
        ASSERT_EQ(legion_acc[*it], SPAN_ACC_BOUNDS_INIT_VAL);
      }
    }

    if (LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)) {
      static constexpr auto EXTENTS = 1000;
      auto exceeded_bounds          = legate::Point<DIM>{EXTENTS};
      const auto legion_acc         = store.read_write_accessor<T, DIM>();
      const auto legion_acc_bounds  = store.read_write_accessor<T, DIM>(bounds);

      ASSERT_EXIT(legion_acc[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(
        legion_acc[(op_shape.hi + legate::Point<DIM>::ONES())], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(legion_acc_bounds[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      ASSERT_EXIT(legion_acc_bounds[(bounds.hi + legate::Point<DIM>::ONES())],
                  ::testing::ExitedWithCode(1),
                  "");
    }
  }
};

class ReadWriteAccessorTestTask : public legate::LegateTask<ReadWriteAccessorTestTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_variant_options(
      legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void ReadWriteAccessorTestTask::cpu_variant(legate::TaskContext context)
{
  auto store = context.output(0).data();

  legate::double_dispatch(store.dim(), store.type().code(), ReadWriteAccessorFn{}, store);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_physical_store_read_write_accessor";

  static void registration_callback(legate::Library library)
  {
    ReadWriteAccessorTestTask::register_variants(library);
  }
};

void test_read_write_accessor_by_task(legate::LogicalStore& logical_store, legate::Scalar& scalar)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, ReadWriteAccessorTestTask::TASK_CONFIG.task_id());

  runtime->issue_fill(logical_store, scalar);
  task.add_input(logical_store);
  task.add_output(logical_store);
  runtime->submit(std::move(task));
}

void test_read_write_accessor_future_store(legate::PhysicalStore& store)
{
  ASSERT_TRUE(store.is_readable());
  ASSERT_FALSE(store.is_writable());
  ASSERT_FALSE(store.is_reducible());
  static constexpr auto DIM = 1;

  ASSERT_EQ(store.shape<DIM>().volume(), DIM);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // accessors of beyond the privilege
    ASSERT_THROW(static_cast<void>(store.read_write_accessor<std::uint64_t, DIM>()),
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

#define PhysicalStoreReadWriteAccessorUnit PhysicalStoreReadWriteAccessorDeathTest

// NOLINTEND(readability-identifier-naming)

#endif

class PhysicalStoreReadWriteAccessorUnit : public RegisterOnceFixture<Config> {};

class BoundStoreReadWriteAccessorTest
  : public PhysicalStoreReadWriteAccessorUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, legate::Type, legate::Scalar>> {
};

// NOLINTBEGIN(readability-magic-numbers)

std::vector<std::tuple<legate::Shape, legate::Type, legate::Scalar>> read_write_accessor_cases()
{
  std::vector<std::tuple<legate::Shape, legate::Type, legate::Scalar>> cases = {
    {legate::Shape{1}, legate::uint32(), legate::Scalar{std::uint32_t{1}}},
    {legate::Shape{2, 3}, legate::bool_(), legate::Scalar{true}},
    {legate::Shape{1, 3, 2}, legate::float16(), legate::Scalar{static_cast<__half>(FLOAT_VALUE)}},
    {legate::Shape{1, 1, 5, 1}, legate::int8(), legate::Scalar{std::int8_t{2}}},
  };

#if LEGATE_MAX_DIM >= 5
  cases.emplace_back(
    legate::Shape{2, 1, 3, 6, 5}, legate::float64(), legate::Scalar{double{FLOAT_VALUE}});
#endif
#if LEGATE_MAX_DIM >= 6
  cases.emplace_back(legate::Shape{1, 2, 3, 4, 5, 6},
                     legate::complex64(),
                     legate::Scalar{complex<float>{FLOAT_VALUE, FLOAT_VALUE}});
#endif
#if LEGATE_MAX_DIM >= 7
  cases.emplace_back(legate::Shape{1, 1, 1, 1, 1, 1, 1},
                     legate::complex128(),
                     legate::Scalar{complex<double>{FLOAT_VALUE, FLOAT_VALUE}});
#endif

  return cases;
}

// NOLINTEND(readability-magic-numbers)

INSTANTIATE_TEST_SUITE_P(
  PhysicalStoreReadWriteAccessorUnit,
  BoundStoreReadWriteAccessorTest,
  ::testing::ValuesIn(read_write_accessor_cases()),
  [](const ::testing::TestParamInfo<BoundStoreReadWriteAccessorTest::ParamType>& param_info) {
    return std::get<legate::Type>(param_info.param).to_string();
  });

std::vector<std::int32_t> generate_axes(std::uint32_t n)
{
  std::vector<std::int32_t> axes(n);
  std::iota(axes.rbegin(), axes.rend(), 0);
  return axes;
}

}  // namespace

TEST_P(BoundStoreReadWriteAccessorTest, BoundStore)
{
  auto [shape, type, scalar] = GetParam();
  auto runtime               = legate::Runtime::get_runtime();
  auto logical_store         = runtime->create_store(shape, type);

  test_read_write_accessor_by_task(logical_store, scalar);
}

TEST_P(BoundStoreReadWriteAccessorTest, TransformedBoundStore)
{
  auto [shape, type, scalar] = GetParam();
  auto runtime               = legate::Runtime::get_runtime();
  auto logical_store         = runtime->create_store(shape, type);
  auto axes                  = generate_axes(shape.ndim());

  logical_store = logical_store.transpose(std::move(axes));
  test_read_write_accessor_by_task(logical_store, scalar);
}

TEST_F(PhysicalStoreReadWriteAccessorUnit, FutureStoreWithTask)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto scalar        = legate::Scalar{UINT64_VALUE};
  auto logical_store = runtime->create_store(legate::Scalar{UINT64_VALUE});

  test_read_write_accessor_by_task(logical_store, scalar);
}

TEST_F(PhysicalStoreReadWriteAccessorUnit, FutureStore)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_store  = runtime->create_store(legate::Scalar{UINT64_VALUE});
  auto physical_store = logical_store.get_physical_store();

  test_read_write_accessor_future_store(physical_store);
}

TEST_F(PhysicalStoreReadWriteAccessorUnit, TransformedFutureStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{UINT64_VALUE});

  logical_store       = logical_store.transpose({0});
  auto physical_store = logical_store.get_physical_store();

  test_read_write_accessor_future_store(physical_store);
}

TEST_F(PhysicalStoreReadWriteAccessorUnit, InvalidDim)
{
  auto runtime                       = legate::Runtime::get_runtime();
  static constexpr auto EXTENTS      = 10;
  auto logical_store                 = runtime->create_store({0, EXTENTS}, legate::int16());
  auto store                         = logical_store.get_physical_store();
  constexpr bool VALIDATE_TYPE       = true;
  constexpr std::int32_t INVALID_DIM = 3;

  ASSERT_THROW(
    static_cast<void>(store.read_write_accessor<std::int16_t, INVALID_DIM, VALIDATE_TYPE>()),
    std::invalid_argument);
  auto bounds = legate::Rect<INVALID_DIM, std::int16_t>({0, 0, 0}, {0, 0, 0});

  ASSERT_THROW(
    static_cast<void>(store.read_write_accessor<std::int16_t, INVALID_DIM, VALIDATE_TYPE>(bounds)),
    std::invalid_argument);
}

TEST_F(PhysicalStoreReadWriteAccessorUnit, InvalidType)
{
  auto runtime                  = legate::Runtime::get_runtime();
  static constexpr auto EXTENTS = 100;
  auto logical_store            = runtime->create_store({0, EXTENTS}, legate::int16());
  auto store                    = logical_store.get_physical_store();
  constexpr bool VALIDATE_TYPE  = true;
  constexpr std::int32_t DIM    = 2;

  ASSERT_THROW(static_cast<void>(store.read_write_accessor<bool, DIM, VALIDATE_TYPE>()),
               std::invalid_argument);
  auto bounds = legate::Rect<DIM, std::uint16_t>{{0, 0}, {0, 0}};

  ASSERT_THROW(static_cast<void>(store.read_write_accessor<double, DIM, VALIDATE_TYPE>(bounds)),
               std::invalid_argument);
}

}  // namespace physical_store_read_write_accessor_test
