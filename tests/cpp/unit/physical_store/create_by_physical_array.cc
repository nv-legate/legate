/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace create_by_physical_array_test {

namespace {

class ArrayStoreFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::PhysicalArray& array) const
  {
    using T                       = legate::type_of_t<CODE>;
    auto store                    = array.data();
    static constexpr auto EXTENTS = 10;

    if (store.is_unbound_store()) {
      static_cast<void>(
        store.create_output_buffer<T, DIM>(legate::Point<DIM>{EXTENTS}, /*bind_buffer=*/true));
    }
    if (array.nullable()) {
      auto null_mask = array.null_mask();

      if (null_mask.is_unbound_store()) {
        static_cast<void>(null_mask.create_output_buffer<bool, DIM>(legate::Point<DIM>{EXTENTS},
                                                                    /*bind_buffer=*/true));
      }
    }

    if (array.nullable()) {
      ASSERT_THROW(static_cast<void>(legate::PhysicalStore{array}), std::invalid_argument);
    } else {
      auto other = legate::PhysicalStore{array};

      ASSERT_EQ(other.dim(), store.dim());
      ASSERT_EQ(other.type().code(), store.type().code());
    }
  }
};

class PrimitiveArrayStoreTask : public legate::LegateTask<PrimitiveArrayStoreTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

/*static*/ void PrimitiveArrayStoreTask::cpu_variant(legate::TaskContext context)
{
  auto array = context.output(0);

  legate::double_dispatch(array.dim(), array.type().code(), ArrayStoreFn{}, array);
}

class ListArrayStoreTask : public legate::LegateTask<ListArrayStoreTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

/*static*/ void ListArrayStoreTask::cpu_variant(legate::TaskContext context)
{
  auto array = context.output(0);

  if (array.nullable()) {
    auto null_mask = array.null_mask();

    if (null_mask.is_unbound_store()) {
      null_mask.bind_empty_data();
    }
  }
  auto list_array       = array.as_list_array();
  auto descriptor_store = list_array.descriptor().data();
  auto vardata_store    = list_array.vardata().data();

  if (descriptor_store.is_unbound_store()) {
    descriptor_store.bind_empty_data();
  }
  ASSERT_NO_THROW(static_cast<void>(
    vardata_store.create_output_buffer<std::int64_t, 1>(legate::Point<1>{10}, true)));
  ASSERT_THROW(static_cast<void>(legate::PhysicalStore{list_array}), std::invalid_argument);
}

class StringArrayStoreTask : public legate::LegateTask<StringArrayStoreTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{3}};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

/*static*/ void StringArrayStoreTask::cpu_variant(legate::TaskContext context)
{
  auto array = context.output(0);

  if (array.nullable()) {
    auto null_mask = array.null_mask();

    if (null_mask.is_unbound_store()) {
      null_mask.bind_empty_data();
    }
  }
  auto string_array = array.as_string_array();
  auto ranges_store = string_array.ranges().data();
  auto chars_store  = string_array.chars().data();

  if (ranges_store.is_unbound_store()) {
    ranges_store.bind_empty_data();
  }
  ASSERT_NO_THROW(static_cast<void>(
    chars_store.create_output_buffer<std::int8_t, 1>(legate::Point<1>{10}, true)));
  ASSERT_THROW(static_cast<void>(legate::PhysicalStore{string_array}), std::invalid_argument);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_create_by_logical_store";

  static void registration_callback(legate::Library library)
  {
    PrimitiveArrayStoreTask::register_variants(library);
    ListArrayStoreTask::register_variants(library);
    StringArrayStoreTask::register_variants(library);
  }
};

class CreateByPhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class CreatePrimitiveUnboundStoreTest
  : public CreateByPhysicalArrayUnit,
    public ::testing::WithParamInterface<
      std::tuple<std::tuple<legate::Type, std::uint32_t>, bool>> {};

class CreatePrimitiveBoundStoreTest : public CreateByPhysicalArrayUnit,
                                      public ::testing::WithParamInterface<
                                        std::tuple<std::tuple<legate::Shape, legate::Type>, bool>> {
};

class OptimizeScalarTest : public CreateByPhysicalArrayUnit,
                           public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(
  CreateByPhysicalArrayUnit,
  CreatePrimitiveBoundStoreTest,
  ::testing::Combine(::testing::Values(std::make_tuple(legate::Shape{1}, legate::int32()),
                                       std::make_tuple(legate::Shape{2, 3}, legate::uint16()),
                                       std::make_tuple(legate::Shape{4, 5, 6}, legate::float16()),
                                       std::make_tuple(legate::Shape{7, 8, 9}, legate::float64())),
                     ::testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(CreateByPhysicalArrayUnit,
                         CreatePrimitiveUnboundStoreTest,
                         ::testing::Combine(::testing::Values(std::make_tuple(legate::int64(), 1),
                                                              std::make_tuple(legate::uint8(), 2),
                                                              std::make_tuple(legate::bool_(), 3),
                                                              std::make_tuple(legate::float32(),
                                                                              4)),
                                            ::testing::Values(true, false)));

INSTANTIATE_TEST_SUITE_P(CreateByPhysicalArrayUnit,
                         OptimizeScalarTest,
                         ::testing::Values(true, false));

void test_array_store(legate::LogicalArray& logical_array, legate::LocalTaskID id)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, id);
  auto part    = task.declare_partition();

  task.add_output(logical_array, part);
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(CreatePrimitiveBoundStoreTest, Basic)
{
  const auto [params, optimize_scalar] = GetParam();
  const auto [shape, type]             = params;
  auto runtime                         = legate::Runtime::get_runtime();
  auto logical_array                   = runtime->create_array(shape, type, optimize_scalar);

  test_array_store(logical_array, PrimitiveArrayStoreTask::TASK_CONFIG.task_id());
}

TEST_P(CreatePrimitiveUnboundStoreTest, Basic)
{
  const auto [params, optimize_scalar] = GetParam();
  const auto [type, dim]               = params;
  auto runtime                         = legate::Runtime::get_runtime();
  auto logical_array                   = runtime->create_array(type, dim, optimize_scalar);

  test_array_store(logical_array, PrimitiveArrayStoreTask::TASK_CONFIG.task_id());
}

TEST_P(OptimizeScalarTest, CreateListBoundArrayStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto list_type     = legate::list_type(legate::int64()).as_list_type();
  auto logical_array = runtime->create_array({2}, list_type, GetParam());

  test_array_store(logical_array, ListArrayStoreTask::TASK_CONFIG.task_id());
}

TEST_P(OptimizeScalarTest, CreateListUnboundArrayStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto list_type     = legate::list_type(legate::int64()).as_list_type();
  auto logical_array = runtime->create_array(list_type, /*dim=*/1, GetParam());

  test_array_store(logical_array, ListArrayStoreTask::TASK_CONFIG.task_id());
}

TEST_P(OptimizeScalarTest, CreateStringBoundArrayStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_array = runtime->create_array({2}, legate::string_type(), GetParam());

  test_array_store(logical_array, StringArrayStoreTask::TASK_CONFIG.task_id());
}

TEST_P(OptimizeScalarTest, CreateStringUnboundArrayStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_array = runtime->create_array(legate::string_type(), /*dim=*/1, GetParam());

  test_array_store(logical_array, StringArrayStoreTask::TASK_CONFIG.task_id());
}

}  // namespace create_by_physical_array_test
