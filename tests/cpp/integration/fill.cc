/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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
              public ::testing::WithParamInterface<std::tuple<std::int32_t, std::size_t>> {};

class Slice : public RegisterOnceFixture<Config>,
              public ::testing::WithParamInterface<std::tuple<bool, std::int32_t>> {};

INSTANTIATE_TEST_SUITE_P(
  FillTests, Whole, ::testing::Combine(::testing::Values(1, 2, 3), ::testing::Values(1, SIZE)));

INSTANTIATE_TEST_SUITE_P(FillTests,
                         Slice,
                         ::testing::Combine(::testing::Bool(), ::testing::Values(1, 2, 3)));

template <std::int32_t DIM>
/*static*/ void CheckTask<DIM>::cpu_variant(legate::TaskContext context)
{
  auto input       = context.input(0);
  auto shape       = input.shape<DIM>();
  const auto value = context.scalar(0).value<std::int64_t>();

  if (shape.empty()) {
    return;
  }

  auto val_acc = input.read_accessor<std::int64_t, DIM>(shape);
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    EXPECT_EQ(val_acc[*it], value);
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

  auto acc         = input.read_accessor<std::int64_t, DIM>(shape);
  auto v_in_slice  = value_in_slice.value<std::int64_t>();
  auto v_out_slice = value_outside_slice.value<std::int64_t>();
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
    EXPECT_EQ(acc[*it], in_slice(*it) ? v_in_slice : v_out_slice);
  }
}

/*static*/ void WrapFillValueTask::cpu_variant(legate::TaskContext context)
{
  auto output        = context.output(0);
  const auto& scalar = context.scalar(0);

  auto acc = output.write_accessor<std::int8_t, 1, false>();
  std::memcpy(acc.ptr(0), scalar.ptr(), scalar.size());
}

void check_output(const legate::LogicalStore& store, const legate::Scalar& value)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto task = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(CHECK_TASK) + store.dim()});
  task.add_input(store);
  task.add_scalar_arg(value);
  runtime->submit(std::move(task));
}

void check_output_slice(const legate::LogicalStore& store,
                        const legate::Scalar& value_in_slice,
                        const legate::Scalar& value_outside_slice,
                        std::int64_t offset)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto task = runtime->create_task(
    context, legate::LocalTaskID{static_cast<std::int64_t>(CHECK_SLICE_TASK) + store.dim()});
  task.add_input(store);
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

void test_fill_index(std::int32_t dim, std::uint64_t size)
{
  auto runtime = legate::Runtime::get_runtime();

  auto lhs = runtime->create_store(legate::full(static_cast<std::uint64_t>(dim), size),
                                   legate::int64(),
                                   /*optimize_scalar=*/true);
  auto v   = legate::Scalar{int64_t{10}};

  // fill input store with some values
  runtime->issue_fill(lhs, v);

  // check the result of fill
  check_output(lhs, v);
}

void test_fill_slice(std::int32_t dim, std::uint64_t size, bool task_init)
{
  auto runtime = legate::Runtime::get_runtime();

  constexpr std::int64_t v1     = 100;
  constexpr std::int64_t v2     = 200;
  constexpr std::int64_t offset = 3;

  auto lhs =
    runtime->create_store(legate::full(static_cast<std::uint64_t>(dim), size), legate::int64());
  auto value_in_slice      = legate::Scalar{v1};
  auto value_outside_slice = legate::Scalar{v2};

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
  auto store   = runtime->create_store(legate::Shape{10, 10}, legate::int64());
  auto v       = legate::Scalar{10.0};

  // Type mismatch
  EXPECT_THROW(runtime->issue_fill(store, runtime->create_store(v)), std::invalid_argument);
  EXPECT_THROW(runtime->issue_fill(store, v), std::invalid_argument);

  // Nulliyfing a (non-nullable) store
  EXPECT_THROW(runtime->issue_fill(store, legate::Scalar{}), std::invalid_argument);
}

}  // namespace

TEST_P(Whole, Index)
{
  const auto& [dim, size] = GetParam();
  test_fill_index(dim, size);
}

TEST_P(Whole, Single)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  const legate::Scope scope{machine.slice(/*from=*/0, /*to=*/1, legate::mapping::TaskTarget::CPU)};

  const auto& [dim, size] = GetParam();
  test_fill_index(dim, size);
}

TEST_P(Slice, Index)
{
  const auto& [task_init, dim] = GetParam();
  test_fill_slice(dim, SIZE, task_init);
}

TEST_F(FillTests, Invalid) { test_invalid(); }

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

// NOLINTEND(readability-magic-numbers)

}  // namespace fill_test
