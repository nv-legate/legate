/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace physical_store_test {

using PhysicalStoreUnit = DefaultFixture;

constexpr uint64_t UINT64_VALUE          = 1;
constexpr uint32_t BOUND_STORE_EXTENTS   = 8;
constexpr uint32_t UNBOUND_STORE_EXTENTS = 9;

static const char* library_name = "legate.physical_store";

enum class UnbountStoreOpCode : uint32_t {
  BIND_EMPTY          = 0,
  BIND_CREATED_BUFFER = 1,
  BIND_BUFFER         = 2,
  INVALID_BINDING     = 3,
  INVALID_DIM         = 4,
  BASIC_FEATURES      = 5
};

enum class AccessorCode : uint32_t {
  READ       = 0,
  WRITE      = 1,
  READ_WRITE = 2,
  REDUCE     = 3,
};

enum class ArrayType : uint32_t {
  PRIMITIVE_ARRAY = 0,
  LIST_ARRAY      = 1,
  STRING_ARRAY    = 2,
  STRUCT_ARRAY    = 3,
};

enum StoreTaskID : int32_t {
  UNBOUND_STORE_TASK_ID         = 0,
  ACCESSOR_TASK_ID              = 1,
  PRIMITIVE_ARRAY_STORE_TASK_ID = 2,
  LIST_ARRAY_STORE_TASK_ID      = 3,
  STRING_ARRAY_STORE_TASK_ID    = 4,
};

template <typename T, int32_t DIM>
void test_unbound_store(const legate::PhysicalStore& store);
template <typename T, int32_t DIM>
void test_bound_store(legate::PhysicalStore& store, const legate::Rect<DIM>& expect_rect);
template <typename T>
void test_future_store(const legate::Scalar& scalar);
legate::PhysicalStore create_unbound_store_by_task(UnbountStoreOpCode op_code,
                                                   const legate::Type& type,
                                                   uint32_t dim = 1);
void test_array_store(legate::LogicalArray& logical_array, StoreTaskID id);
void register_tasks();
legate::Shape get_shape(int32_t dim);
void test_RO_accessor(legate::LogicalStore& logical_store);
void test_WO_accessor(legate::LogicalStore& logical_store);
void test_RW_accessor(legate::LogicalStore& logical_store);
void test_RD_accessor(legate::LogicalStore& logical_store);
template <int32_t DIM>
void test_accessors_normal_store();
void test_accessor_future_store();
void test_invalid_accessor();

struct unbound_store_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore& store, uint32_t op_code)
  {
    using T                 = legate::type_of<CODE>;
    UnbountStoreOpCode code = static_cast<UnbountStoreOpCode>(op_code);
    switch (code) {
      case UnbountStoreOpCode::BIND_EMPTY: {
        store.bind_empty_data();
        break;
      }
      case UnbountStoreOpCode::BIND_CREATED_BUFFER: {
        auto buffer =
          store.create_output_buffer<T, DIM>(legate::Point<DIM>{UNBOUND_STORE_EXTENTS}, true);
        break;
      }
      case UnbountStoreOpCode::BIND_BUFFER: {
        auto buffer = store.create_output_buffer<T, DIM>(legate::Point<DIM>{UNBOUND_STORE_EXTENTS});
        store.bind_data(buffer, legate::Point<DIM>::ONES());
        break;
      }
      case UnbountStoreOpCode::INVALID_BINDING: {
        auto buffer =
          store.create_output_buffer<T, DIM>(legate::Point<DIM>{UNBOUND_STORE_EXTENTS}, true);
        EXPECT_THROW(store.bind_data(buffer, legate::Point<DIM>::ONES()), std::invalid_argument);
        EXPECT_THROW(store.bind_empty_data(), std::invalid_argument);
        break;
      }
      case UnbountStoreOpCode::INVALID_DIM: {
        constexpr int32_t INVALID_DIM = DIM % LEGATE_MAX_DIM + 1;
        EXPECT_THROW(static_cast<void>(
                       store.create_output_buffer<T>(legate::Point<INVALID_DIM>::ONES(), true)),
                     std::invalid_argument);

        // bind to buffer
        store.bind_empty_data();
        break;
      }
      case UnbountStoreOpCode::BASIC_FEATURES: {
        test_unbound_store<T, DIM>(store);

        // bind to buffer
        store.bind_empty_data();
        break;
      }
    }
  }
};

struct read_accessor_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore& store, int32_t value)
  {
    EXPECT_TRUE(store.is_readable());
    EXPECT_FALSE(store.is_writable());
    EXPECT_FALSE(store.is_reducible());

    using T       = legate::type_of<CODE>;
    auto read_acc = store.read_accessor<T, DIM>();
    auto op_shape = store.shape<DIM>();
    if (!op_shape.empty()) {
      for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
        EXPECT_EQ(read_acc[*it], static_cast<T>(value));
      }
    }

    auto bounds = legate::Rect<DIM>{op_shape.lo + legate::Point<DIM>::ONES(), op_shape.hi};
    if (bounds.empty()) {
      return;
    }

    for (legate::PointInRectIterator<DIM> it{bounds}; it.valid(); ++it) {
      EXPECT_EQ(read_acc[*it], static_cast<T>(value));
    }

    if (LegateDefined(LEGATE_BOUNDS_CHECKS)) {
      // access store with exceeded bounds
      auto read_acc_bounds = store.read_accessor<T, DIM>(bounds);
      auto exceeded_bounds = legate::Point<DIM>(10000);
      EXPECT_EXIT(read_acc[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(
        read_acc[(op_shape.hi + legate::Point<DIM>(100))], ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(read_acc_bounds[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(read_acc_bounds[(bounds.hi + legate::Point<DIM>::ONES())],
                  ::testing::ExitedWithCode(1),
                  "");
    }

    if (LegateDefined(LEGATE_USE_DEBUG)) {
      // accessors of beyond the privilege
      EXPECT_THROW(static_cast<void>(store.write_accessor<T, DIM>()), std::invalid_argument);
      EXPECT_THROW(static_cast<void>(store.read_write_accessor<T, DIM>()), std::invalid_argument);
      EXPECT_THROW(static_cast<void>(store.reduce_accessor<legate::SumReduction<T>, false, DIM>()),
                   std::invalid_argument);
    }
  }
};

struct write_accessor_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore& store)
  {
    EXPECT_TRUE(store.is_readable());
    EXPECT_TRUE(store.is_writable());
    EXPECT_TRUE(store.is_reducible());

    using T        = legate::type_of<CODE>;
    auto write_acc = store.write_accessor<T, DIM>();
    auto op_shape  = store.shape<DIM>();
    if (!op_shape.empty()) {
      for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
        write_acc[*it] = static_cast<T>(2);
        EXPECT_EQ(write_acc[*it], static_cast<T>(2));
      }
    }

    auto bounds = legate::Rect<DIM>{op_shape.lo + legate::Point<DIM>::ONES(), op_shape.hi};
    if (bounds.empty()) {
      return;
    }

    auto write_acc_bounds = store.write_accessor<T, DIM>(bounds);
    for (legate::PointInRectIterator<DIM> it{bounds}; it.valid(); ++it) {
      write_acc_bounds[*it] = static_cast<T>(4);
      EXPECT_EQ(write_acc_bounds[*it], static_cast<T>(4));
    }

    if (LegateDefined(LEGATE_BOUNDS_CHECKS)) {
      auto exceeded_bounds = legate::Point<DIM>(10000);
      EXPECT_EXIT(write_acc[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(
        write_acc[(op_shape.hi + legate::Point<DIM>::ONES())], ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(write_acc_bounds[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(write_acc_bounds[(bounds.hi + legate::Point<DIM>::ONES())],
                  ::testing::ExitedWithCode(1),
                  "");
    }
  }
};

struct read_write_accessor_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalStore& store)
  {
    EXPECT_TRUE(store.is_readable());
    EXPECT_TRUE(store.is_writable());
    EXPECT_TRUE(store.is_reducible());

    using T             = legate::type_of<CODE>;
    auto read_write_acc = store.read_write_accessor<T, DIM>();
    auto op_shape       = store.shape<DIM>();
    if (!op_shape.empty()) {
      for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
        read_write_acc[*it] = static_cast<T>(5);
        EXPECT_EQ(read_write_acc[*it], static_cast<T>(5));
      }
    }

    auto bounds = legate::Rect<DIM>{op_shape.lo + legate::Point<DIM>::ONES(), op_shape.hi};
    if (bounds.empty()) {
      return;
    }

    auto read_write_acc_bounds = store.read_write_accessor<T, DIM>(bounds);
    for (legate::PointInRectIterator<DIM> it{bounds}; it.valid(); ++it) {
      read_write_acc_bounds[*it] = static_cast<T>(6);
      EXPECT_EQ(read_write_acc_bounds[*it], static_cast<T>(6));
    }

    if (LegateDefined(LEGATE_BOUNDS_CHECKS)) {
      auto exceeded_bounds = legate::Point<DIM>(10000);
      EXPECT_EXIT(read_write_acc[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(read_write_acc[(op_shape.hi + legate::Point<DIM>::ONES())],
                  ::testing::ExitedWithCode(1),
                  "");
      EXPECT_EXIT(read_write_acc_bounds[exceeded_bounds], ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(read_write_acc_bounds[(bounds.hi + legate::Point<DIM>::ONES())],
                  ::testing::ExitedWithCode(1),
                  "");
    }
  }
};

struct reduce_accessor_fn {
  template <int32_t DIM>
  void operator()(legate::PhysicalStore& store)
  {
    EXPECT_TRUE(store.is_readable());
    EXPECT_TRUE(store.is_writable());
    EXPECT_TRUE(store.is_reducible());

    auto reduce_acc = store.reduce_accessor<legate::SumReduction<int64_t>, false, DIM>();
    auto op_shape   = store.shape<DIM>();
    if (!op_shape.empty()) {
      for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
        legate::Point<DIM> pos{*it};
        reduce_acc.reduce(pos, 10);
      }
    }

    auto bounds = legate::Rect<DIM>{op_shape.lo + legate::Point<DIM>::ONES(), op_shape.hi};
    if (bounds.empty()) {
      return;
    }

    auto reduce_acc_bounds =
      store.reduce_accessor<legate::SumReduction<int64_t>, false, DIM>(bounds);
    for (legate::PointInRectIterator<DIM> it{bounds}; it.valid(); ++it) {
      legate::Point<DIM> pos(*it);
      reduce_acc_bounds.reduce(pos, 10);
    }

    if (LegateDefined(LEGATE_BOUNDS_CHECKS)) {
      auto exceeded_bounds = legate::Point<DIM>(10000);
      EXPECT_EXIT(reduce_acc.reduce(exceeded_bounds, 10), ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(reduce_acc.reduce((op_shape.hi + legate::Point<DIM>::ONES()), 10),
                  ::testing::ExitedWithCode(1),
                  "");
      EXPECT_EXIT(reduce_acc_bounds.reduce(exceeded_bounds, 10), ::testing::ExitedWithCode(1), "");
      EXPECT_EXIT(reduce_acc_bounds.reduce((bounds.hi + legate::Point<DIM>::ONES()), 10),
                  ::testing::ExitedWithCode(1),
                  "");
    }
  }
};

struct UnboundStoreTask : public legate::LegateTask<UnboundStoreTask> {
  static const int32_t TASK_ID = StoreTaskID::UNBOUND_STORE_TASK_ID;
  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void UnboundStoreTask::cpu_variant(legate::TaskContext context)
{
  auto store   = context.output(0).data();
  auto op_code = context.scalar(0).value<uint32_t>();
  legate::double_dispatch(store.dim(), store.code(), unbound_store_fn{}, store, op_code);
}

struct AccessorTestTask : public legate::LegateTask<AccessorTestTask> {
  static const int32_t TASK_ID = StoreTaskID::ACCESSOR_TASK_ID;
  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void AccessorTestTask::cpu_variant(legate::TaskContext context)
{
  auto op_code = context.scalar(0).value<uint32_t>();

  AccessorCode code = static_cast<AccessorCode>(op_code);
  switch (code) {
    case AccessorCode::READ: {
      auto value = context.scalar(1).value<int64_t>();
      auto store = context.input(0).data();
      legate::double_dispatch(store.dim(), store.type().code(), read_accessor_fn{}, store, value);
      break;
    }
    case AccessorCode::WRITE: {
      auto store = context.output(0).data();
      legate::double_dispatch(store.dim(), store.type().code(), write_accessor_fn{}, store);
      break;
    }
    case AccessorCode::READ_WRITE: {
      auto store = context.output(0).data();
      legate::double_dispatch(store.dim(), store.type().code(), read_write_accessor_fn{}, store);
      break;
    }
    case AccessorCode::REDUCE: {
      auto store = context.reduction(0).data();
      legate::dim_dispatch(store.dim(), reduce_accessor_fn{}, store);
      break;
    }
  }
}

struct PrimitiveArrayStoreTask : public legate::LegateTask<PrimitiveArrayStoreTask> {
  static const int32_t TASK_ID = StoreTaskID::PRIMITIVE_ARRAY_STORE_TASK_ID;
  static void cpu_variant(legate::TaskContext context);
};

struct array_store_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::PhysicalArray& array)
  {
    using T    = legate::type_of<CODE>;
    auto store = array.data();
    if (store.is_unbound_store()) {
      static_cast<void>(store.create_output_buffer<T, DIM>(legate::Point<DIM>{10}, true));
    }
    if (array.nullable()) {
      auto null_mask = array.null_mask();
      if (null_mask.is_unbound_store()) {
        static_cast<void>(null_mask.create_output_buffer<bool, DIM>(legate::Point<DIM>{10}, true));
      }
    }

    if (!array.nullable()) {
      auto other = legate::PhysicalStore{array};
      EXPECT_EQ(other.dim(), store.dim());
      EXPECT_EQ(other.type().code(), store.type().code());
    } else {
      EXPECT_THROW(static_cast<void>(legate::PhysicalStore{array}), std::invalid_argument);
    }
  }
};

/*static*/ void PrimitiveArrayStoreTask::cpu_variant(legate::TaskContext context)
{
  auto array = context.output(0);
  legate::double_dispatch(array.dim(), array.type().code(), array_store_fn{}, array);
}

struct ListArrayStoreTask : public legate::LegateTask<ListArrayStoreTask> {
  static const int32_t TASK_ID = StoreTaskID::LIST_ARRAY_STORE_TASK_ID;
  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void ListArrayStoreTask::cpu_variant(legate::TaskContext context)
{
  auto array            = context.output(0);
  auto list_array       = array.as_list_array();
  auto descriptor_store = list_array.descriptor().data();
  auto vardata_store    = list_array.vardata().data();
  auto buffer = vardata_store.create_output_buffer<int64_t, 1>(legate::Point<1>{10}, true);
  if (array.nullable()) {
    auto null_mask = array.null_mask();
    if (null_mask.is_unbound_store()) {
      null_mask.bind_empty_data();
    }
  }
  if (descriptor_store.is_unbound_store()) {
    descriptor_store.bind_empty_data();
  }

  EXPECT_THROW(static_cast<void>(legate::PhysicalStore{list_array}), std::invalid_argument);
}

struct StringArrayStoreTask : public legate::LegateTask<StringArrayStoreTask> {
  static const int32_t TASK_ID = StoreTaskID::STRING_ARRAY_STORE_TASK_ID;
  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void StringArrayStoreTask::cpu_variant(legate::TaskContext context)
{
  auto array        = context.output(0);
  auto string_array = array.as_string_array();
  auto ranges_store = string_array.ranges().data();
  auto chars_store  = string_array.chars().data();
  auto buffer       = chars_store.create_output_buffer<int8_t, 1>(legate::Point<1>{10}, true);
  if (ranges_store.is_unbound_store()) {
    ranges_store.bind_empty_data();
  }

  if (array.nullable()) {
    auto null_mask = array.null_mask();
    if (null_mask.is_unbound_store()) {
      null_mask.bind_empty_data();
    }
  }

  EXPECT_THROW(static_cast<void>(legate::PhysicalStore{string_array}), std::invalid_argument);
}

void test_RO_accessor(legate::LogicalStore& logical_store)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto context        = runtime->find_library(library_name);
  const int64_t value = 0;
  runtime->issue_fill(logical_store, legate::Scalar{value});
  auto task = runtime->create_task(context, StoreTaskID::ACCESSOR_TASK_ID);
  task.add_input(logical_store);
  task.add_scalar_arg(legate::Scalar{static_cast<uint32_t>(AccessorCode::READ)});
  task.add_scalar_arg(legate::Scalar{value});
  runtime->submit(std::move(task));
}

void test_WO_accessor(legate::LogicalStore& logical_store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, StoreTaskID::ACCESSOR_TASK_ID);
  task.add_output(logical_store);
  task.add_scalar_arg(legate::Scalar{static_cast<uint32_t>(AccessorCode::WRITE)});
  runtime->submit(std::move(task));
}

void test_RW_accessor(legate::LogicalStore& logical_store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, StoreTaskID::ACCESSOR_TASK_ID);
  task.add_input(logical_store);
  task.add_output(logical_store);
  task.add_scalar_arg(legate::Scalar{static_cast<uint32_t>(AccessorCode::READ_WRITE)});
  runtime->submit(std::move(task));
}

void test_RD_accessor(legate::LogicalStore& logical_store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, StoreTaskID::ACCESSOR_TASK_ID);
  task.add_reduction(logical_store, legate::ReductionOpKind::ADD);
  task.add_scalar_arg(legate::Scalar{static_cast<uint32_t>(AccessorCode::REDUCE)});
  runtime->submit(std::move(task));
}

template <int32_t DIM>
void test_accessors_normal_store()
{
  auto runtime       = legate::Runtime::get_runtime();
  auto extents       = get_shape(DIM);
  auto logical_store = runtime->create_store(extents, legate::int64());
  test_RO_accessor(logical_store);
  test_WO_accessor(logical_store);
  test_RW_accessor(logical_store);
  test_RD_accessor(logical_store);
}

void test_accessor_future_store()
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{UINT64_VALUE});
  auto store         = logical_store.get_physical_store();

  // Note: gitlab issue #10: future wrappers are read-only now.
  EXPECT_TRUE(store.is_readable());
  EXPECT_FALSE(store.is_writable());
  EXPECT_FALSE(store.is_reducible());

  EXPECT_EQ(store.shape<1>().volume(), 1);

  auto read_acc = store.read_accessor<uint64_t, 1>();
  EXPECT_EQ(read_acc[0], UINT64_VALUE);
  EXPECT_EQ(read_acc[0], store.scalar<uint64_t>());

  if (LegateDefined(LEGATE_BOUNDS_CHECKS)) {
    // access store with exceeded bounds
    auto exceeded_bounds = legate::Point<1>{1000};
    EXPECT_EXIT(read_acc[exceeded_bounds], ::testing::ExitedWithCode(1), "");
  }
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    // accessors of beyond the privilege
    EXPECT_THROW(static_cast<void>(store.write_accessor<uint64_t, 1>()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.read_write_accessor<uint64_t, 1>()),
                 std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(store.reduce_accessor<legate::SumReduction<u_int64_t>, false, 1>()),
      std::invalid_argument);
  }
}

void test_invalid_accessor()
{
  // invalid dim
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store({10, 20}, legate::int16());
    auto store         = logical_store.get_physical_store();

    constexpr int32_t INVALID_DIM = 3;
    EXPECT_THROW(static_cast<void>(store.read_accessor<int16_t, INVALID_DIM>()),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.write_accessor<int16_t, INVALID_DIM>()),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.read_write_accessor<int16_t, INVALID_DIM>()),
                 std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(store.reduce_accessor<legate::SumReduction<int16_t>, true, INVALID_DIM>()),
      std::invalid_argument);

    auto bounds = legate::Rect<INVALID_DIM, int16_t>({0, 0, 0}, {0, 0, 0});
    EXPECT_THROW(static_cast<void>(store.read_accessor<int16_t, INVALID_DIM>(bounds)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.write_accessor<int16_t, INVALID_DIM>(bounds)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.read_write_accessor<int16_t, INVALID_DIM>(bounds)),
                 std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(
        store.reduce_accessor<legate::SumReduction<int16_t>, false, INVALID_DIM>(bounds)),
      std::invalid_argument);
  }

  // invalid type
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store({5, 6}, legate::int16());
    auto store         = logical_store.get_physical_store();

    constexpr int32_t DIM = 2;
    EXPECT_THROW(static_cast<void>(store.read_accessor<int32_t, DIM>()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.write_accessor<uint64_t, DIM>()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.read_write_accessor<bool, DIM>()), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(store.reduce_accessor<legate::SumReduction<int8_t>, true, DIM>()),
      std::invalid_argument);

    auto bounds = legate::Rect<DIM, uint16_t>{{0, 0}, {0, 0}};
    EXPECT_THROW(static_cast<void>(store.read_accessor<uint32_t, DIM>(bounds)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.write_accessor<int64_t, DIM>(bounds)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.read_write_accessor<double, DIM>(bounds)),
                 std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(store.reduce_accessor<legate::SumReduction<float>, false, DIM>(bounds)),
      std::invalid_argument);
  }
}

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  UnboundStoreTask::register_variants(context);
  AccessorTestTask::register_variants(context);
  PrimitiveArrayStoreTask::register_variants(context);
  ListArrayStoreTask::register_variants(context);
  StringArrayStoreTask::register_variants(context);
}

legate::Shape get_shape(int32_t dim)
{
  if (dim < 0) {
    return legate::Shape{0};
  }
  std::vector<size_t> vec;
  vec.reserve(dim);
  for (int32_t i = 0; i < dim; i++) {
    vec.emplace_back(BOUND_STORE_EXTENTS + i);
  }
  return legate::Shape{std::move(vec)};
}

template <typename T, int32_t DIM>
void test_unbound_store(const legate::PhysicalStore& store)
{
  EXPECT_FALSE(store.is_future());
  EXPECT_TRUE(store.is_unbound_store());
  EXPECT_EQ(store.dim(), DIM);
  EXPECT_TRUE(store.valid());
  EXPECT_EQ(store.type().code(), legate::type_code_of<T>);
  EXPECT_EQ(store.code(), legate::type_code_of<T>);
  EXPECT_FALSE(store.transformed());

  EXPECT_THROW(static_cast<void>(store.shape<DIM>()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(store.domain()), std::invalid_argument);

  // Specfic APIs for future/bound store
  EXPECT_THROW(static_cast<void>(store.scalar<T>()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(store.read_accessor<T, DIM>()), std::invalid_argument);
}

legate::PhysicalStore create_unbound_store_by_task(UnbountStoreOpCode op_code,
                                                   const legate::Type& type,
                                                   uint32_t dim)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto logical_store = runtime->create_store(type, dim);
  auto task          = runtime->create_task(context, StoreTaskID::UNBOUND_STORE_TASK_ID);
  task.add_output(logical_store);
  task.add_scalar_arg(legate::Scalar(static_cast<uint32_t>(op_code)));
  runtime->submit(std::move(task));

  // Turns out to be a bound store here
  auto store = logical_store.get_physical_store();
  EXPECT_FALSE(store.is_unbound_store());
  EXPECT_FALSE(logical_store.unbound());
  return store;
}

template <typename T>
void test_future_store(const legate::Scalar& scalar)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(scalar);
  auto store         = logical_store.get_physical_store();

  constexpr int32_t DIM = 1;
  auto expect_rect      = legate::Rect<DIM>{0, 0};

  EXPECT_TRUE(store.is_future());
  EXPECT_FALSE(store.is_unbound_store());
  EXPECT_EQ(store.dim(), DIM);
  EXPECT_TRUE(store.valid());
  EXPECT_EQ(store.type().code(), legate::type_code_of<T>);
  EXPECT_EQ(store.code(), legate::type_code_of<T>);
  EXPECT_FALSE(store.transformed());

  EXPECT_EQ(store.shape<DIM>(), expect_rect);
  EXPECT_THROW(static_cast<void>(store.shape<2>()), std::invalid_argument);

  auto domain = store.domain();
  EXPECT_EQ(domain.get_dim(), DIM);
  auto actual_rect = domain.bounds<DIM, T>();
  EXPECT_EQ(actual_rect, expect_rect);

  // Specfic API for future store
  EXPECT_EQ(store.scalar<T>(), scalar.value<T>());

  // Specfic APIs for bound/unbound stores
  EXPECT_THROW(static_cast<void>(store.create_output_buffer<T>(legate::Point<DIM>::ONES())),
               std::invalid_argument);
  EXPECT_THROW(store.bind_empty_data(), std::invalid_argument);
}

template <typename T, int32_t DIM>
void test_bound_store(legate::PhysicalStore& store, const legate::Rect<DIM>& expect_rect)
{
  EXPECT_FALSE(store.is_future());
  EXPECT_FALSE(store.is_unbound_store());
  EXPECT_EQ(store.dim(), DIM);
  EXPECT_TRUE(store.valid());
  EXPECT_EQ(store.type().code(), legate::type_code_of<T>);
  EXPECT_EQ(store.code(), legate::type_code_of<T>);
  EXPECT_FALSE(store.transformed());

  EXPECT_EQ(store.shape<DIM>(), expect_rect);
  constexpr int32_t INVALID_DIM = std::max((DIM + 1) % LEGATE_MAX_DIM, 1);
  EXPECT_THROW(static_cast<void>(store.shape<INVALID_DIM>()), std::invalid_argument);

  auto domain = store.domain();
  EXPECT_EQ(domain.get_dim(), DIM);
  auto actual_rect = domain.bounds<DIM, size_t>();
  EXPECT_EQ(actual_rect, expect_rect);

  // Specfic API for future/unbound store
  EXPECT_THROW(static_cast<void>(store.scalar<T>()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(store.create_output_buffer<T>(legate::Point<DIM>::ONES())),
               std::invalid_argument);
  EXPECT_THROW(store.bind_empty_data(), std::invalid_argument);
}

void test_array_store(legate::LogicalArray& logical_array, StoreTaskID id)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, id);
  auto part    = task.declare_partition();
  task.add_output(logical_array, part);
  runtime->submit(std::move(task));
}

TEST_F(PhysicalStoreUnit, FutureStoreCreation)
{
  test_future_store<uint64_t>(legate::Scalar{UINT64_VALUE});
}

TEST_F(PhysicalStoreUnit, FutureStoreInvalid)
{
  auto runtime = legate::Runtime::get_runtime();
  EXPECT_THROW(static_cast<void>(runtime->create_store(legate::Scalar{1}, {3})),
               std::invalid_argument);
}

TEST_F(PhysicalStoreUnit, BoundStoreMultiDims)
{
  auto runtime = legate::Runtime::get_runtime();
#if LEGATE_MAX_DIM >= 1
  {
    constexpr int32_t DIM = 1;
    auto logical_store    = runtime->create_store({5}, legate::int64());
    auto store            = logical_store.get_physical_store();
    legate::Rect<DIM> expect_rect{0, 4};
    test_bound_store<int64_t, DIM>(store, expect_rect);
  }
#endif
#if LEGATE_MAX_DIM >= 2
  {
    constexpr int32_t DIM = 2;
    auto logical_store    = runtime->create_store({6, 5}, legate::bool_());
    auto store            = logical_store.get_physical_store();
    legate::Rect<DIM> expect_rect{{0, 0}, {5, 4}};
    test_bound_store<bool, DIM>(store, expect_rect);
  }
#endif
#if LEGATE_MAX_DIM >= 3
  {
    constexpr int32_t DIM = 3;
    auto logical_store    = runtime->create_store({100, 10, 1}, legate::float16());
    auto store            = logical_store.get_physical_store();
    legate::Rect<DIM> expect_rect{{0, 0, 0}, {99, 9, 0}};
    test_bound_store<__half, DIM>(store, expect_rect);
  }
#endif
#if LEGATE_MAX_DIM >= 4
  {
    constexpr int32_t DIM = 4;
    auto logical_store    = runtime->create_store({7, 100, 8, 1000}, legate::complex128());
    auto store            = logical_store.get_physical_store();
    legate::Rect<DIM> expect_rect{{0, 0, 0, 0}, {6, 99, 7, 999}};
    test_bound_store<complex<double>, DIM>(store, expect_rect);
  }
#endif
#if LEGATE_MAX_DIM >= 5
  {
    constexpr int32_t DIM = 5;
    auto logical_store    = runtime->create_store({20, 6, 4, 10, 50}, legate::uint16());
    auto store            = logical_store.get_physical_store();
    std::vector<legate::coord_t> lo{0, 0, 0, 0, 0};
    std::vector<legate::coord_t> hi{19, 5, 3, 9, 49};
    legate::Rect<DIM> expect_rect{legate::Point<5>{lo.data()}, legate::Point<5>{hi.data()}};
    test_bound_store<uint16_t, DIM>(store, expect_rect);
  }
#endif
#if LEGATE_MAX_DIM >= 6
  {
    constexpr int32_t DIM = 6;
    auto logical_store    = runtime->create_store({1, 2, 3, 4, 5, 6}, legate::float64());
    auto store            = logical_store.get_physical_store();
    std::vector<legate::coord_t> lo{0, 0, 0, 0, 0, 0};
    std::vector<legate::coord_t> hi{0, 1, 2, 3, 4, 5};
    legate::Rect<DIM> expect_rect{legate::Point<6>{lo.data()}, legate::Point<6>{hi.data()}};
    test_bound_store<double, DIM>(store, expect_rect);
  }
#endif
#if LEGATE_MAX_DIM >= 7
  {
    constexpr int32_t DIM = 7;
    auto logical_store    = runtime->create_store({7, 6, 5, 4, 3, 2, 1}, legate::complex64());
    auto store            = logical_store.get_physical_store();
    std::vector<legate::coord_t> lo = {0, 0, 0, 0, 0, 0, 0};
    std::vector<legate::coord_t> hi = {6, 5, 4, 3, 2, 1, 0};
    legate::Rect<DIM> expect_rect{legate::Point<7>{lo.data()}, legate::Point<7>{hi.data()}};
    test_bound_store<complex<float>, DIM>(store, expect_rect);
  }
#endif
}

TEST_F(PhysicalStoreUnit, BoundStoreEmptyShape)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({}, legate::int64());
  auto store         = logical_store.get_physical_store();
  EXPECT_EQ(store.dim(), 0);
}

TEST_F(PhysicalStoreUnit, BoundStoreOptimizeScalar)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_store1 = runtime->create_store({1}, legate::int64(), true);
  auto store1         = logical_store1.get_physical_store();
  EXPECT_TRUE(store1.is_future());

  auto logical_store2 = runtime->create_store({1, 2}, legate::int64(), true);
  auto store2         = logical_store2.get_physical_store();
  EXPECT_FALSE(store2.is_future());

  auto logical_store3 = runtime->create_store({1}, legate::int64(), false);
  auto store3         = logical_store3.get_physical_store();
  EXPECT_FALSE(store3.is_future());
}

TEST_F(PhysicalStoreUnit, BoundStoreInvalid)
{
  constexpr int32_t DIM = 2;
  auto runtime          = legate::Runtime::get_runtime();
  auto logical_store    = runtime->create_store({static_cast<size_t>(-2), 1}, legate::int64());
  auto store            = logical_store.get_physical_store();
  auto expect_rect      = legate::Rect<DIM, int64_t>{{0, 0}, {-3, 0}};
  test_bound_store<int64_t, DIM>(store, expect_rect);
}

TEST_F(PhysicalStoreUnit, UnboundStoreCreation)
{
  register_tasks();
#if LEGATE_MAX_DIM >= 1
  create_unbound_store_by_task(UnbountStoreOpCode::BASIC_FEATURES, legate::uint32(), 1);
#endif
#if LEGATE_MAX_DIM >= 2
  create_unbound_store_by_task(UnbountStoreOpCode::BASIC_FEATURES, legate::bool_(), 2);
#endif
#if LEGATE_MAX_DIM >= 3
  create_unbound_store_by_task(UnbountStoreOpCode::BASIC_FEATURES, legate::float16(), 3);
#endif
#if LEGATE_MAX_DIM >= 4
  create_unbound_store_by_task(UnbountStoreOpCode::BASIC_FEATURES, legate::float32(), 4);
#endif
#if LEGATE_MAX_DIM >= 5
  create_unbound_store_by_task(UnbountStoreOpCode::BASIC_FEATURES, legate::float64(), 5);
#endif
#if LEGATE_MAX_DIM >= 6
  create_unbound_store_by_task(UnbountStoreOpCode::BASIC_FEATURES, legate::complex64(), 6);
#endif
#if LEGATE_MAX_DIM >= 7
  create_unbound_store_by_task(UnbountStoreOpCode::BASIC_FEATURES, legate::complex128(), 7);
#endif
}

TEST_F(PhysicalStoreUnit, UnboundStoreBindBuffer)
{
  register_tasks();
  constexpr int32_t DIM = 1;

  {
    auto store = create_unbound_store_by_task(UnbountStoreOpCode::BIND_EMPTY, legate::int32());
    // empty rect
    auto expect_rect = legate::Rect<DIM, int32_t>{0, -1};
    test_bound_store<int32_t, DIM>(store, expect_rect);
  }

  {
    auto store =
      create_unbound_store_by_task(UnbountStoreOpCode::BIND_CREATED_BUFFER, legate::float32());
    auto expect_rect = legate::Rect<DIM, int32_t>{0, UNBOUND_STORE_EXTENTS - 1};
    test_bound_store<float, DIM>(store, expect_rect);
  }

  {
    auto store = create_unbound_store_by_task(UnbountStoreOpCode::BIND_BUFFER, legate::complex64());
    // equals extents of binded buffer
    auto expect_rect = legate::Rect<DIM, int32_t>{0, 0};
    test_bound_store<complex<float>, DIM>(store, expect_rect);
  }
}

TEST_F(PhysicalStoreUnit, UnboundStoreInvalid)
{
  register_tasks();
  create_unbound_store_by_task(UnbountStoreOpCode::INVALID_BINDING, legate::int64());
  create_unbound_store_by_task(UnbountStoreOpCode::INVALID_DIM, legate::uint8());
}

TEST_F(PhysicalStoreUnit, PrimitiveArrayStoreCreation)
{
  register_tasks();
  auto runtime = legate::Runtime::get_runtime();

  // Bound
  {
    auto logical_array1 = runtime->create_array({2, 4}, legate::int64(), false);
    test_array_store(logical_array1, StoreTaskID::PRIMITIVE_ARRAY_STORE_TASK_ID);

    auto logical_array2 = runtime->create_array({2, 4}, legate::int64(), true);
    test_array_store(logical_array2, StoreTaskID::PRIMITIVE_ARRAY_STORE_TASK_ID);
  }

  // Unbound
  {
    auto logical_array1 = runtime->create_array(legate::int64(), 2, false);
    test_array_store(logical_array1, StoreTaskID::PRIMITIVE_ARRAY_STORE_TASK_ID);

    auto logical_array2 = runtime->create_array(legate::int64(), 2, true);
    test_array_store(logical_array2, StoreTaskID::PRIMITIVE_ARRAY_STORE_TASK_ID);
  }
}

TEST_F(PhysicalStoreUnit, ListArrayStoreCreation)
{
  register_tasks();
  auto runtime   = legate::Runtime::get_runtime();
  auto list_type = legate::list_type(legate::int64()).as_list_type();

  // Bound
  {
    auto logical_array1 = runtime->create_array({2}, list_type, false);
    test_array_store(logical_array1, StoreTaskID::LIST_ARRAY_STORE_TASK_ID);

    auto logical_array2 = runtime->create_array({2}, list_type, true);
    test_array_store(logical_array2, StoreTaskID::LIST_ARRAY_STORE_TASK_ID);
  }

  // Unbound
  {
    auto logical_array1 = runtime->create_array(list_type, 1, false);
    test_array_store(logical_array1, StoreTaskID::LIST_ARRAY_STORE_TASK_ID);

    auto logical_array2 = runtime->create_array(list_type, 1, true);
    test_array_store(logical_array2, StoreTaskID::LIST_ARRAY_STORE_TASK_ID);
  }
}

TEST_F(PhysicalStoreUnit, StringArrayStoreCreation)
{
  register_tasks();
  auto runtime  = legate::Runtime::get_runtime();
  auto str_type = legate::string_type();

  // Bound
  {
    auto logical_array1 = runtime->create_array({2}, str_type, false);
    test_array_store(logical_array1, StoreTaskID::STRING_ARRAY_STORE_TASK_ID);

    auto logical_array2 = runtime->create_array({2}, str_type, true);
    test_array_store(logical_array2, StoreTaskID::STRING_ARRAY_STORE_TASK_ID);
  }

  // Unbound
  {
    auto logical_array1 = runtime->create_array(str_type, 1, false);
    test_array_store(logical_array1, StoreTaskID::STRING_ARRAY_STORE_TASK_ID);

    auto logical_array2 = runtime->create_array(str_type, 1, true);
    test_array_store(logical_array2, StoreTaskID::STRING_ARRAY_STORE_TASK_ID);
  }
}

TEST_F(PhysicalStoreUnit, StoreCreationLike)
{
  // Bound Store
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store({2, 3}, legate::int64());
    auto store         = logical_store.get_physical_store();
    legate::PhysicalStore other1{store};
    EXPECT_EQ(other1.dim(), store.dim());
    EXPECT_EQ(other1.type().code(), store.type().code());
    EXPECT_EQ(other1.shape<2>(), store.shape<2>());

    legate::PhysicalStore other2{logical_store.get_physical_store()};
    EXPECT_EQ(other2.dim(), store.dim());
    EXPECT_EQ(other2.type().code(), store.type().code());
    EXPECT_EQ(other2.shape<2>(), store.shape<2>());
  }

  // Future Store
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store(legate::Scalar{UINT64_VALUE});
    auto store         = logical_store.get_physical_store();
    legate::PhysicalStore other1{store};
    EXPECT_EQ(other1.dim(), store.dim());
    EXPECT_EQ(other1.type().code(), store.type().code());
    EXPECT_EQ(other1.shape<1>(), store.shape<1>());

    legate::PhysicalStore other2{logical_store.get_physical_store()};
    EXPECT_EQ(other2.dim(), store.dim());
    EXPECT_EQ(other2.type().code(), store.type().code());
    EXPECT_EQ(other2.shape<1>(), store.shape<1>());
  }
}

TEST_F(PhysicalStoreUnit, Assignment)
{
  // Bound Store
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store({2, 3}, legate::int64());
    auto store         = logical_store.get_physical_store();
    auto other1        = store;
    EXPECT_EQ(other1.dim(), store.dim());
    EXPECT_EQ(other1.type().code(), store.type().code());
    EXPECT_EQ(other1.shape<2>(), store.shape<2>());

    auto other2 = logical_store.get_physical_store();
    EXPECT_EQ(other2.dim(), store.dim());
    EXPECT_EQ(other2.type().code(), store.type().code());
    EXPECT_EQ(other2.shape<2>(), store.shape<2>());
  }

  // Future Store
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store(legate::Scalar{UINT64_VALUE});
    auto store         = logical_store.get_physical_store();
    auto other1        = store;
    EXPECT_EQ(other1.dim(), store.dim());
    EXPECT_EQ(other1.type().code(), store.type().code());
    EXPECT_EQ(other1.shape<1>(), store.shape<1>());

    auto other2 = logical_store.get_physical_store();
    EXPECT_EQ(other2.dim(), store.dim());
    EXPECT_EQ(other2.type().code(), store.type().code());
    EXPECT_EQ(other2.shape<1>(), store.shape<1>());
  }
}

TEST_F(PhysicalStoreUnit, Transform)
{
  // future store
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store(legate::Scalar{UINT64_VALUE});
    auto store         = logical_store.get_physical_store();
    EXPECT_FALSE(store.transformed());

    auto promoted = logical_store.promote(0, 1);
    store         = promoted.get_physical_store();
    EXPECT_TRUE(store.transformed());
  }

  // bound store
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store({1, 7}, legate::int64());
    auto store         = logical_store.get_physical_store();
    EXPECT_FALSE(store.transformed());

    auto promoted = logical_store.promote(0, 1);
    store         = promoted.get_physical_store();
    EXPECT_TRUE(store.transformed());
  }
}

TEST_F(PhysicalStoreUnit, NormalStoreAccessor)
{
  register_tasks();
#if LEGATE_MAX_DIM >= 1
  test_accessors_normal_store<1>();
#endif
#if LEGATE_MAX_DIM >= 2
  test_accessors_normal_store<2>();
#endif
#if LEGATE_MAX_DIM >= 3
  test_accessors_normal_store<3>();
#endif
#if LEGATE_MAX_DIM >= 4
  test_accessors_normal_store<4>();
#endif
#if LEGATE_MAX_DIM >= 5
  test_accessors_normal_store<5>();
#endif
#if LEGATE_MAX_DIM >= 6
  test_accessors_normal_store<6>();
#endif
#if LEGATE_MAX_DIM >= 7
  test_accessors_normal_store<7>();
#endif
}

TEST_F(PhysicalStoreUnit, FutureStoreAccessor) { test_accessor_future_store(); }

TEST_F(PhysicalStoreUnit, InvalidAccessor) { test_invalid_accessor(); }
}  // namespace physical_store_test
