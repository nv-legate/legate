/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utilities/utilities.h>

namespace fill_test {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::size_t SIZE = 10;

enum TaskIDs : std::uint8_t {
  CHECK_TASK         = 0,
  CHECK_SLICE_TASK   = 3,
  WRAP_FILL_VAL_TASK = 7,
};

template <std::int32_t DIM>
struct CheckTask : public legate::LegateTask<CheckTask<DIM>> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CHECK_TASK + DIM}};
  static void cpu_variant(legate::TaskContext context);
};

template <std::int32_t DIM>
struct CheckSliceTask : public legate::LegateTask<CheckSliceTask<DIM>> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CHECK_SLICE_TASK + DIM}};
  static void cpu_variant(legate::TaskContext context);
};

struct WrapFillValueTask : public legate::LegateTask<WrapFillValueTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{WRAP_FILL_VAL_TASK}};
  static void cpu_variant(legate::TaskContext context);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_fill";

  static void registration_callback(legate::Library library)
  {
    CheckTask<1>::register_variants(library);
    CheckTask<2>::register_variants(library);
    CheckTask<3>::register_variants(library);
    CheckSliceTask<1>::register_variants(library);
    CheckSliceTask<2>::register_variants(library);
    CheckSliceTask<3>::register_variants(library);
    WrapFillValueTask::register_variants(library);
  }
};

class FillTests : public RegisterOnceFixture<Config> {};

class Whole : public RegisterOnceFixture<Config>,
              public ::testing::WithParamInterface<std::tuple<bool, std::int32_t, std::size_t>> {};

class Slice : public RegisterOnceFixture<Config>,
              public ::testing::WithParamInterface<std::tuple<bool, bool, std::int32_t>> {};

INSTANTIATE_TEST_SUITE_P(FillTests,
                         Whole,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Values(1, 2, 3),
                                            ::testing::Values(1, SIZE)));

INSTANTIATE_TEST_SUITE_P(FillTests,
                         Slice,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(1, 2, 3)));

template <std::int32_t DIM>
/*static*/ void CheckTask<DIM>::cpu_variant(legate::TaskContext context)
{
  auto input           = context.input(0);
  auto shape           = input.shape<DIM>();
  const auto value     = context.scalar(0).value<std::int64_t>();
  const auto null_mask = context.scalar(1).value<bool>();

  if (shape.empty()) {
    return;
  }

  auto val_acc = input.data().read_accessor<std::int64_t, DIM>(shape);
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    EXPECT_EQ(val_acc[*it], value);
  }

  if (!input.nullable()) {
    return;
  }

  auto mask_acc = input.null_mask().read_accessor<bool, DIM>(shape);
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    EXPECT_EQ(mask_acc[*it], null_mask);
  }
}

template <std::int32_t DIM>
/*static*/ void CheckSliceTask<DIM>::cpu_variant(legate::TaskContext context)
{
  auto input                      = context.input(0);
  auto shape                      = input.shape<DIM>();
  const auto& value_in_slice      = context.scalar(0);
  const auto& value_outside_slice = context.scalar(1);
  auto offset                     = context.scalar(2).value<std::int64_t>();

  if (shape.empty()) {
    return;
  }

  auto in_slice = [&offset](const auto& p) {
    for (std::int32_t dim = 0; dim < DIM; ++dim) {
      if (p[dim] < offset) {
        return false;
      }
    }
    return true;
  };

  if (!input.nullable()) {
    auto acc         = input.data().read_accessor<std::int64_t, DIM>(shape);
    auto v_in_slice  = value_in_slice.value<std::int64_t>();
    auto v_out_slice = value_outside_slice.value<std::int64_t>();
    for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
      EXPECT_EQ(acc[*it], in_slice(*it) ? v_in_slice : v_out_slice);
    }
    return;
  }

  auto val_acc    = input.data().read_accessor<std::int64_t, DIM>(shape);
  auto mask_acc   = input.null_mask().read_accessor<bool, DIM>(shape);
  auto v_in_slice = value_in_slice.value<std::int64_t>();
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    if (in_slice(*it)) {
      EXPECT_EQ(val_acc[*it], v_in_slice);
      EXPECT_EQ(mask_acc[*it], true);
    } else {
      EXPECT_EQ(mask_acc[*it], false);
    }
  }
}

/*static*/ void WrapFillValueTask::cpu_variant(legate::TaskContext context)
{
  auto output        = context.output(0);
  const auto& scalar = context.scalar(0);

  auto acc = output.data().write_accessor<std::int8_t, 1, false>();
  std::memcpy(acc.ptr(0), scalar.ptr(), scalar.size());
}

void check_output(const legate::LogicalArray& array, const legate::Scalar& value, bool null_mask)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto task = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(CHECK_TASK) + array.dim()});
  task.add_input(array);
  task.add_scalar_arg(value);
  task.add_scalar_arg(legate::Scalar{null_mask});
  runtime->submit(std::move(task));
}

void check_output_slice(const legate::LogicalArray& array,
                        const legate::Scalar& value_in_slice,
                        const legate::Scalar& value_outside_slice,
                        std::int64_t offset)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto task = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(CHECK_SLICE_TASK) + array.dim()});
  task.add_input(array);
  task.add_scalar_arg(value_in_slice);
  task.add_scalar_arg(value_outside_slice);
  task.add_scalar_arg(legate::Scalar{offset});
  runtime->submit(std::move(task));
}

legate::LogicalStore wrap_fill_value(const legate::Scalar& value)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto result  = runtime->create_store(legate::Shape{1}, value.type(), /*optimize_scalar=*/true);

  auto task = runtime->create_task(context, legate::LocalTaskID{WRAP_FILL_VAL_TASK});
  task.add_output(result);
  task.add_scalar_arg(value);
  runtime->submit(std::move(task));

  return result;
}

void test_fill_index(std::int32_t dim, std::uint64_t size, bool nullable)
{
  auto runtime = legate::Runtime::get_runtime();

  auto lhs = runtime->create_array(legate::full(static_cast<std::uint64_t>(dim), size),
                                   legate::int64(),
                                   nullable /*nullable*/,
                                   /*optimize_scalar=*/true);
  auto v   = legate::Scalar{int64_t{10}};

  // fill input array with some values
  runtime->issue_fill(lhs, v);

  // check the result of fill
  check_output(lhs, v, /*null_mask*/ true);
}

void test_fill_slice(std::int32_t dim, std::uint64_t size, bool null_init, bool task_init)
{
  auto runtime = legate::Runtime::get_runtime();

  constexpr std::int64_t v1     = 100;
  constexpr std::int64_t v2     = 200;
  constexpr std::int64_t offset = 3;

  auto lhs = runtime->create_array(
    legate::full(static_cast<std::uint64_t>(dim), size), legate::int64(), null_init);
  auto value_in_slice      = legate::Scalar{v1};
  auto value_outside_slice = null_init ? legate::Scalar{} : legate::Scalar{v2};

  // First fill the entire store with v2
  runtime->issue_fill(lhs, value_outside_slice);

  // Then fill a slice with v1
  auto sliced = lhs;
  for (std::int32_t idx = 0; idx < dim; ++idx) {
    sliced = sliced.slice(idx, legate::Slice{offset});
  }
  if (task_init) {
    runtime->issue_fill(sliced, wrap_fill_value(value_in_slice));
  } else {
    runtime->issue_fill(sliced, value_in_slice);
  }

  // check if the slice is filled correctly
  check_output_slice(lhs, value_in_slice, value_outside_slice, offset);
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::Shape{10, 10}, legate::int64(), /*nullable=*/false);
  auto v       = legate::Scalar{10.0};

  // Type mismatch
  EXPECT_THROW(runtime->issue_fill(array, runtime->create_store(v)), std::invalid_argument);
  EXPECT_THROW(runtime->issue_fill(array, v), std::invalid_argument);

  // Nulliyfing a non-nullable array
  EXPECT_THROW(runtime->issue_fill(array, legate::Scalar{}), std::invalid_argument);
}

}  // namespace

TEST_P(Whole, Index)
{
  const auto& [nullable, dim, size] = GetParam();
  test_fill_index(dim, size, nullable);
}

TEST_P(Whole, Single)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  const legate::Scope scope{machine.slice(/*from=*/0, /*to=*/1, legate::mapping::TaskTarget::CPU)};

  const auto& [nullable, dim, size] = GetParam();
  test_fill_index(dim, size, nullable);
}

TEST_P(Slice, Index)
{
  const auto& [null_init, task_init, dim] = GetParam();
  test_fill_slice(dim, SIZE, null_init, task_init);
}

TEST_F(FillTests, Invalid) { test_invalid(); }

TEST_F(FillTests, FillNullableArrayWithScalar)
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::Shape{10, 10}, legate::int64(), /*nullable*/ true);
  const auto value = legate::Scalar{};

  runtime->issue_fill(array, value);
  check_output(array, legate::Scalar{int64_t{0}}, /*null_mask*/ false);
}

TEST_F(FillTests, FillNullableArrayWithStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::Shape{10, 10}, legate::int64(), /*nullable*/ true);
  const auto value = runtime->create_store(legate::null_type());

  runtime->issue_fill(array, value);
  check_output(array, legate::Scalar{int64_t{0}}, /*null_mask*/ false);
}

TEST_F(FillTests, FillUnboundStoreWithScalar)
{
  auto runtime     = legate::Runtime::get_runtime();
  auto store       = runtime->create_store(legate::int64());
  const auto value = legate::Scalar{10};

  ASSERT_THAT([&] { runtime->issue_fill(store, value); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Fill lhs cannot be an unbound store")));
}

TEST_F(FillTests, FillUnboundStoreWithStore)
{
  auto runtime     = legate::Runtime::get_runtime();
  auto store       = runtime->create_store(legate::int64());
  const auto value = runtime->create_store(legate::int64());

  ASSERT_THAT([&] { runtime->issue_fill(store, value); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Fill lhs cannot be an unbound store")));
}

TEST_F(FillTests, FillStructArrayWithScalar)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto struct_array = runtime->create_array(legate::struct_type(/*align=*/true, legate::int64()));
  const auto value  = legate::Scalar{10};

  ASSERT_THAT([&] { runtime->issue_fill(struct_array, value); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Fills on list or struct arrays are not supported yet")));
}

TEST_F(FillTests, FillStructArrayWithStore)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto struct_array = runtime->create_array(legate::struct_type(/*align=*/true, legate::int64()));
  const auto value  = runtime->create_store(legate::int64());

  ASSERT_THAT([&] { runtime->issue_fill(struct_array, value); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Fills on list or struct arrays are not supported yet")));
}

TEST_F(FillTests, FillArrayWithNullScalar)
{
  auto runtime     = legate::Runtime::get_runtime();
  auto array       = runtime->create_array(legate::int64());
  const auto value = legate::Scalar{};

  ASSERT_THAT([&] { runtime->issue_fill(array, value); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nullable arrays cannot be filled with null")));
}

TEST_F(FillTests, FillArrayWithNullStore)
{
  auto runtime     = legate::Runtime::get_runtime();
  auto array       = runtime->create_array(legate::int64());
  const auto value = runtime->create_store(legate::null_type());

  ASSERT_THAT([&] { runtime->issue_fill(array, value); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nullable arrays cannot be filled with null")));
}

// NOLINTEND(readability-magic-numbers)

}  // namespace fill_test
