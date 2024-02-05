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

namespace physical_array_unit_test {

using PhysicalArrayUnit = DefaultFixture;

static const char* library_name = "legate.physical_array";

enum class ArrayTaskID : int32_t {
  PRIMITIVE_UNBOUND_ARRAY_TASK_ID = 0,
  LIST_ARRAY_TASK_ID              = 1,
  STRING_ARRAY_TASK_ID            = 2,
  FILL_ARRAY_TASK_ID              = 3,
  CHECK_ARRAY_TASK_ID             = 4,
};

enum class ArrayType : int32_t {
  PRIMITIVE_ARRAY = 0,
  LIST_ARRAY      = 1,
  STRING_TYPE     = 2,
};

struct UnboundArrayTask : public legate::LegateTask<UnboundArrayTask> {
  static constexpr int32_t TASK_ID =
    static_cast<std::underlying_type_t<ArrayTaskID>>(ArrayTaskID::PRIMITIVE_UNBOUND_ARRAY_TASK_ID);
  static void cpu_variant(legate::TaskContext context);
};

struct ListArrayTask : public legate::LegateTask<ListArrayTask> {
  static constexpr int32_t TASK_ID =
    static_cast<std::underlying_type_t<ArrayTaskID>>(ArrayTaskID::LIST_ARRAY_TASK_ID);
  static void cpu_variant(legate::TaskContext context);
};

struct StringArrayTask : public legate::LegateTask<StringArrayTask> {
  static constexpr int32_t TASK_ID =
    static_cast<std::underlying_type_t<ArrayTaskID>>(ArrayTaskID::STRING_ARRAY_TASK_ID);
  static void cpu_variant(legate::TaskContext context);
};

struct FillTask : public legate::LegateTask<FillTask> {
  static constexpr int32_t TASK_ID =
    static_cast<std::underlying_type_t<ArrayTaskID>>(ArrayTaskID::FILL_ARRAY_TASK_ID);
  static void cpu_variant(legate::TaskContext context);
};

struct CheckTask : public legate::LegateTask<CheckTask> {
  static constexpr int32_t TASK_ID =
    static_cast<std::underlying_type_t<ArrayTaskID>>(ArrayTaskID::CHECK_ARRAY_TASK_ID);
  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void UnboundArrayTask::cpu_variant(legate::TaskContext context)
{
  auto array                   = context.output(0);
  auto nullable                = context.scalar(0).value<bool>();
  auto store                   = array.data();
  static constexpr int32_t DIM = 3;
  EXPECT_TRUE(store.is_unbound_store());
  ASSERT_NO_THROW(
    static_cast<void>(store.create_output_buffer<uint32_t, DIM>(legate::Point<DIM>(10), true)));

  EXPECT_EQ(array.nullable(), nullable);
  EXPECT_EQ(array.dim(), DIM);
  EXPECT_EQ(array.type(), legate::uint32());
  EXPECT_FALSE(array.nested());
  EXPECT_THROW(static_cast<void>(array.shape<DIM>()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(array.domain()), std::invalid_argument);

  EXPECT_TRUE(store.is_unbound_store());
  EXPECT_FALSE(store.is_future());
  EXPECT_EQ(store.dim(), DIM);
  EXPECT_THROW(static_cast<void>(store.shape<DIM>()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(store.domain()), std::invalid_argument);
  EXPECT_EQ(store.type(), legate::uint32());

  if (nullable) {
    auto null_mask = array.null_mask();
    EXPECT_TRUE(null_mask.is_unbound_store());
    ASSERT_NO_THROW(
      static_cast<void>(null_mask.create_output_buffer<bool, DIM>(legate::Point<DIM>(10), true)));
    EXPECT_THROW(static_cast<void>(null_mask.shape<DIM>()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(null_mask.domain()), std::invalid_argument);
    EXPECT_EQ(null_mask.type(), legate::bool_());
    EXPECT_EQ(null_mask.dim(), array.dim());
  } else {
    EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  EXPECT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(array.as_list_array()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(array.as_string_array()), std::invalid_argument);
}

template <typename T, int32_t DIM>
void test_array_data(legate::PhysicalStore& store, bool is_unbound, legate::Type::Code code)
{
  EXPECT_EQ(store.is_unbound_store(), is_unbound);
  EXPECT_EQ(store.dim(), DIM);
  EXPECT_EQ(store.type().code(), code);
  if (is_unbound) {
    EXPECT_THROW(static_cast<void>(store.shape<1>()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(store.domain()), std::invalid_argument);
  }
}

/*static*/ void ListArrayTask::cpu_variant(legate::TaskContext context)
{
  auto array                   = context.output(0);
  auto nullable                = context.scalar(0).value<bool>();
  auto unbound                 = context.scalar(1).value<bool>();
  static constexpr int32_t DIM = 1;
  auto list_array              = array.as_list_array();
  auto descriptor_store        = list_array.descriptor().data();
  auto vardata_store           = list_array.vardata().data();
  auto buffer = vardata_store.create_output_buffer<int64_t, DIM>(legate::Point<1>(100), true);
  if (unbound) {
    ASSERT_NO_THROW(descriptor_store.bind_empty_data());
  }

  EXPECT_EQ(array.nullable(), nullable);
  EXPECT_EQ(array.dim(), DIM);
  EXPECT_EQ(array.type().code(), legate::list_type(legate::int64()).code());
  EXPECT_TRUE(array.nested());
  if (unbound) {
    EXPECT_THROW(static_cast<void>(array.shape<DIM>()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(array.domain()), std::invalid_argument);
  }

  if (nullable) {
    auto null_mask = array.null_mask();
    if (null_mask.is_unbound_store()) {
      ASSERT_NO_THROW(null_mask.bind_empty_data());
      EXPECT_THROW(static_cast<void>(null_mask.shape<DIM>()), std::invalid_argument);
      EXPECT_THROW(static_cast<void>(null_mask.domain()), std::invalid_argument);
    }
    EXPECT_EQ(null_mask.type(), legate::bool_());
    EXPECT_EQ(null_mask.dim(), array.dim());
  } else {
    EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  test_array_data<int64_t, DIM>(descriptor_store, unbound, legate::Type::Code::STRUCT);
  test_array_data<int64_t, DIM>(vardata_store, true, legate::Type::Code::INT64);

  auto desc = array.child(0).data();
  auto var  = array.child(1).data();
  test_array_data<int64_t, DIM>(desc, unbound, legate::Type::Code::STRUCT);
  test_array_data<int64_t, DIM>(var, true, legate::Type::Code::INT64);

  // invalid
  EXPECT_THROW(static_cast<void>(array.child(2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(array.child(-1)), std::out_of_range);

  EXPECT_THROW(static_cast<void>(array.as_string_array()), std::invalid_argument);
}

/*static*/ void StringArrayTask::cpu_variant(legate::TaskContext context)
{
  auto array    = context.output(0);
  auto nullable = context.scalar(0).value<bool>();
  auto unbound  = context.scalar(1).value<bool>();

  auto string_array            = array.as_string_array();
  auto ranges_store            = string_array.ranges().data();
  auto chars_store             = string_array.chars().data();
  static constexpr int32_t DIM = 1;
  ASSERT_NO_THROW(
    static_cast<void>(chars_store.create_output_buffer<int8_t, DIM>(legate::Point<DIM>(10), true)));
  if (unbound) {
    ASSERT_NO_THROW(ranges_store.bind_empty_data());
  }

  EXPECT_EQ(array.nullable(), nullable);
  EXPECT_EQ(array.dim(), DIM);
  EXPECT_EQ(array.type(), legate::string_type());
  EXPECT_TRUE(array.nested());
  if (unbound) {
    EXPECT_THROW(static_cast<void>(array.shape<DIM>()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(array.domain()), std::invalid_argument);
  }

  if (nullable) {
    auto null_mask = array.null_mask();
    if (null_mask.is_unbound_store()) {
      ASSERT_NO_THROW(null_mask.bind_empty_data());
      EXPECT_THROW(static_cast<void>(null_mask.shape<DIM>()), std::invalid_argument);
      EXPECT_THROW(static_cast<void>(null_mask.domain()), std::invalid_argument);
    }
    EXPECT_EQ(null_mask.type(), legate::bool_());
    EXPECT_EQ(null_mask.dim(), array.dim());
  } else {
    EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  test_array_data<int8_t, DIM>(ranges_store, unbound, legate::Type::Code::STRUCT);
  test_array_data<int8_t, DIM>(chars_store, true, legate::Type::Code::INT8);

  auto ranges = array.child(0).data();
  auto chars  = array.child(1).data();
  test_array_data<int8_t, DIM>(ranges, unbound, legate::Type::Code::STRUCT);
  test_array_data<int8_t, DIM>(chars, true, legate::Type::Code::INT8);

  // cast to ListArray
  auto list_array       = array.as_list_array();
  auto descriptor_store = list_array.descriptor().data();
  auto vardata_store    = list_array.vardata().data();
  test_array_data<int8_t, DIM>(descriptor_store, unbound, legate::Type::Code::STRUCT);
  test_array_data<int8_t, DIM>(vardata_store, true, legate::Type::Code::INT8);

  // invalid
  EXPECT_THROW(static_cast<void>(array.child(2)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(array.child(-1)), std::out_of_range);
}

void fill_bound_base_array(legate::PhysicalArray& array, bool nullable)
{
  auto store                   = array.data();
  static constexpr int32_t DIM = 2;
  auto w_store                 = store.write_accessor<int32_t, DIM>();
  auto store_shape             = store.shape<DIM>();
  auto i                       = 0;
  if (!store_shape.empty()) {
    for (legate::PointInRectIterator<2> it{store_shape}; it.valid(); ++it) {
      w_store[*it] = i;
      i++;
    }
  }

  if (nullable) {
    auto null_mask  = array.null_mask();
    auto w_mask     = null_mask.write_accessor<bool, DIM>();
    auto mask_shape = null_mask.shape<DIM>();
    auto index      = 0;
    if (!mask_shape.empty()) {
      for (legate::PointInRectIterator<2> it{mask_shape}; it.valid(); ++it) {
        w_mask[*it] = (index % 2 == 0);
        index++;
      }
    }
  }
}

void fill_unbound_base_array(legate::PhysicalArray& array, bool nullable)
{
  auto store = array.data();
  ASSERT_NO_THROW(
    static_cast<void>(store.create_output_buffer<int32_t, 2>(legate::Point<2>(5), true)));
  if (nullable) {
    auto null_mask = array.null_mask();
    ASSERT_NO_THROW(
      static_cast<void>(null_mask.create_output_buffer<bool, 2>(legate::Point<2>(5), true)));
  }
}

void check_bound_base_array(legate::PhysicalArray& array, bool nullable)
{
  auto store                   = array.data();
  static constexpr int32_t DIM = 2;
  auto r_store                 = store.read_accessor<int32_t, DIM>();
  auto store_shape             = store.shape<DIM>();
  auto i                       = 0;
  if (!store_shape.empty()) {
    for (legate::PointInRectIterator<DIM> it{store_shape}; it.valid(); ++it) {
      EXPECT_EQ(r_store[*it], i);
      i++;
    }
  }

  if (nullable) {
    auto null_mask  = array.null_mask();
    auto r_mask     = null_mask.read_accessor<bool, DIM>();
    auto mask_shape = null_mask.shape<DIM>();
    auto index      = 0;
    if (!mask_shape.empty()) {
      for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
        EXPECT_EQ(r_mask[*it], (index % 2 == 0));
        index++;
      }
    }
  }
}

void check_unbound_base_array(legate::PhysicalArray& array, bool nullable)
{
  auto store                   = array.data();
  static constexpr int32_t DIM = 2;
  auto rw_store                = store.read_write_accessor<int32_t, DIM>();
  auto store_shape             = store.shape<DIM>();
  auto i                       = 2;
  if (!store_shape.empty()) {
    for (legate::PointInRectIterator<DIM> it(store_shape); it.valid(); ++it) {
      rw_store[*it] = i;
      i++;
    }
    i = 2;
    for (legate::PointInRectIterator<DIM> it(store_shape); it.valid(); ++it) {
      EXPECT_EQ(rw_store[*it], i);
      i++;
    }
  }

  if (nullable) {
    auto null_mask  = array.null_mask();
    auto r_mask     = null_mask.read_write_accessor<bool, DIM>();
    auto mask_shape = null_mask.shape<DIM>();
    auto index      = 0;
    if (!mask_shape.empty()) {
      for (legate::PointInRectIterator<DIM> it(mask_shape); it.valid(); ++it) {
        r_mask[*it] = (index % 2 == 0);
        index++;
      }
      index = 0;
      for (legate::PointInRectIterator<DIM> it(mask_shape); it.valid(); ++it) {
        EXPECT_EQ(r_mask[*it], (index % 2 == 0));
        index++;
      }
    }
  }
}

void bind_list_array(legate::PhysicalArray& array, bool nullable, bool unbound)
{
  auto list_array              = array.as_list_array();
  auto descriptor_store        = list_array.descriptor().data();
  auto vardata_store           = list_array.vardata().data();
  static constexpr int32_t DIM = 1;
  ASSERT_NO_THROW(static_cast<void>(
    vardata_store.create_output_buffer<int64_t, DIM>(legate::Point<DIM>(10), true)));
  if (unbound) {
    ASSERT_NO_THROW(descriptor_store.bind_empty_data());
  }

  if (nullable) {
    auto null_mask = array.null_mask();
    if (null_mask.is_unbound_store()) {
      ASSERT_NO_THROW(null_mask.bind_empty_data());
    }
  }
}

void check_list_array(legate::PhysicalArray& array, bool nullable, bool unbound)
{
  auto list_array              = array.as_list_array();
  auto vardata_store           = list_array.vardata().data();
  static constexpr int32_t DIM = 1;
  auto rw_vardata              = vardata_store.read_write_accessor<int64_t, DIM>();
  auto vardata_shape           = vardata_store.shape<DIM>();
  auto i                       = 0;
  if (!vardata_shape.empty()) {
    for (legate::PointInRectIterator<1> it{vardata_shape}; it.valid(); ++it) {
      rw_vardata[*it] = i;
      i++;
    }
    i = 0;
    for (legate::PointInRectIterator<1> it{vardata_shape}; it.valid(); ++it) {
      EXPECT_EQ(rw_vardata[*it], i);
      i++;
    }
  }

  if (nullable) {
    auto null_mask = array.null_mask();
    if (!unbound) {
      auto rw_mask    = null_mask.read_write_accessor<bool, DIM>();
      auto mask_shape = null_mask.shape<DIM>();
      auto index      = 0;
      if (!mask_shape.empty()) {
        for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
          rw_mask[*it] = (index % 2 == 0);
          index++;
        }
        index = 0;
        for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
          EXPECT_EQ(rw_mask[*it], (index % 2 == 0));
          index++;
        }
      }
    }
  }
}

void bind_string_array(legate::PhysicalArray& array, bool nullable, bool unbound)
{
  auto string_array            = array.as_string_array();
  auto ranges_store            = string_array.ranges().data();
  auto chars_store             = string_array.chars().data();
  static constexpr int32_t DIM = 1;
  ASSERT_NO_THROW(
    static_cast<void>(chars_store.create_output_buffer<int8_t, 1>(legate::Point<DIM>(10), true)));
  if (unbound) {
    ASSERT_NO_THROW(ranges_store.bind_empty_data());
  }

  if (nullable) {
    auto null_mask = array.null_mask();
    if (null_mask.is_unbound_store()) {
      ASSERT_NO_THROW(null_mask.bind_empty_data());
    }
  }
}

void check_string_array(legate::PhysicalArray& array, bool nullable, bool unbound)
{
  auto string_array            = array.as_string_array();
  auto chars_store             = string_array.chars().data();
  static constexpr int32_t DIM = 1;
  auto rw_chars                = chars_store.read_write_accessor<int8_t, DIM>();
  auto chars_shape             = chars_store.shape<1>();
  auto i                       = 0;
  if (!chars_shape.empty()) {
    for (legate::PointInRectIterator<1> it{chars_shape}; it.valid(); ++it) {
      rw_chars[*it] = i;
      i++;
    }
    i = 0;
    for (legate::PointInRectIterator<1> it{chars_shape}; it.valid(); ++it) {
      EXPECT_EQ(rw_chars[*it], i);
      i++;
    }
  }

  if (nullable) {
    auto null_mask = array.null_mask();
    if (!unbound) {
      auto rw_mask    = null_mask.read_write_accessor<bool, 1>();
      auto mask_shape = null_mask.shape<1>();
      auto index      = 0;
      if (!mask_shape.empty()) {
        for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
          rw_mask[*it] = (index % 2 == 0);
          index++;
        }
        index = 0;
        for (legate::PointInRectIterator<DIM> it{mask_shape}; it.valid(); ++it) {
          EXPECT_EQ(rw_mask[*it], (index % 2 == 0));
          index++;
        }
      }
    }
  }
}

/*static*/ void FillTask::cpu_variant(legate::TaskContext context)
{
  auto array    = context.output(0);
  auto nullable = context.scalar(0).value<bool>();
  auto kind     = context.scalar(1).value<std::underlying_type_t<ArrayType>>();
  auto unbound  = context.scalar(2).value<bool>();

  switch (static_cast<ArrayType>(kind)) {
    case ArrayType::PRIMITIVE_ARRAY: {
      if (unbound) {
        fill_unbound_base_array(array, nullable);
      } else {
        fill_bound_base_array(array, nullable);
      }
      break;
    }
    case ArrayType::LIST_ARRAY: {
      bind_list_array(array, nullable, unbound);
      break;
    }
    case ArrayType::STRING_TYPE: {
      bind_string_array(array, nullable, unbound);
      break;
    }
  }
}

/*static*/ void CheckTask::cpu_variant(legate::TaskContext context)
{
  auto array    = context.output(0);
  auto nullable = context.scalar(0).value<bool>();
  auto kind     = context.scalar(1).value<std::underlying_type_t<ArrayType>>();
  auto unbound  = context.scalar(2).value<bool>();

  switch (static_cast<ArrayType>(kind)) {
    case ArrayType::PRIMITIVE_ARRAY: {
      if (unbound) {
        check_unbound_base_array(array, nullable);
      } else {
        check_bound_base_array(array, nullable);
      }
      break;
    }
    case ArrayType::LIST_ARRAY: {
      check_list_array(array, nullable, unbound);
      break;
    }
    case ArrayType::STRING_TYPE: {
      check_string_array(array, nullable, unbound);
      break;
    }
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
  UnboundArrayTask::register_variants(context);
  ListArrayTask::register_variants(context);
  StringArrayTask::register_variants(context);
  FillTask::register_variants(context);
  CheckTask::register_variants(context);
}

void test_bound_array(bool nullable)
{
  auto runtime                 = legate::Runtime::get_runtime();
  auto logical_array           = runtime->create_array({2, 4}, legate::int64(), nullable);
  auto array                   = logical_array.get_physical_array();
  static constexpr int32_t DIM = 2;
  EXPECT_EQ(array.nullable(), nullable);
  EXPECT_EQ(array.dim(), DIM);
  EXPECT_EQ(array.type(), legate::int64());
  EXPECT_FALSE(array.nested());
  EXPECT_EQ(array.shape<DIM>(), legate::Rect<2>({0, 0}, {1, 3}));
  EXPECT_EQ((array.domain().bounds<DIM, int64_t>()), legate::Rect<DIM>({0, 0}, {1, 3}));

  auto store = array.data();
  EXPECT_FALSE(store.is_unbound_store());
  EXPECT_FALSE(store.is_future());
  EXPECT_EQ(store.dim(), DIM);
  EXPECT_EQ(store.shape<DIM>(), legate::Rect<2>({0, 0}, {1, 3}));
  EXPECT_EQ(store.type(), legate::int64());

  if (!nullable) {
    EXPECT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  } else {
    auto null_mask = array.null_mask();
    EXPECT_EQ(null_mask.shape<2>(), array.shape<2>());
    EXPECT_EQ(null_mask.domain(), array.domain());
    EXPECT_EQ(null_mask.type(), legate::bool_());
    EXPECT_EQ(null_mask.dim(), array.dim());
  }
  EXPECT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(array.as_list_array()), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(array.as_string_array()), std::invalid_argument);
}

void test_unbound_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto dim           = 3;
  auto logical_array = runtime->create_array(legate::uint32(), dim, nullable);
  auto task          = runtime->create_task(
    context, static_cast<int64_t>(ArrayTaskID::PRIMITIVE_UNBOUND_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  runtime->submit(std::move(task));

  EXPECT_FALSE(logical_array.unbound());
}

void test_bound_list_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto arr_type      = legate::list_type(legate::int64()).as_list_type();
  auto logical_array = runtime->create_array({6}, arr_type, nullable);
  auto task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::LIST_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{false});
  runtime->submit(std::move(task));
}

void test_unbound_list_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto arr_type      = legate::list_type(legate::int64()).as_list_type();
  auto dim           = 1;
  auto logical_array = runtime->create_array(arr_type, dim, nullable);
  auto task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::LIST_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{true});
  runtime->submit(std::move(task));
}

void test_bound_string_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto str_type      = legate::string_type();
  auto logical_array = runtime->create_array({5}, str_type, nullable);
  auto task =
    runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::STRING_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{false});
  runtime->submit(std::move(task));
}

void test_unbound_string_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto str_type      = legate::string_type();
  auto dim           = 1;
  auto logical_array = runtime->create_array(str_type, dim, nullable);
  auto task =
    runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::STRING_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{true});
  runtime->submit(std::move(task));
}

void test_primitive_array(bool nullable)
{
  register_tasks();
  test_bound_array(nullable);
  test_unbound_array(nullable);
}

void test_list_array(bool nullable)
{
  register_tasks();
  test_bound_list_array(nullable);
  test_unbound_list_array(nullable);
}

void test_string_array(bool nullable)
{
  register_tasks();
  test_bound_string_array(nullable);
  test_unbound_string_array(nullable);
}

void test_fill_bound_primitive_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto logical_array = runtime->create_array({1, 4}, legate::int32(), nullable);

  // Fill task
  auto task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::FILL_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::PRIMITIVE_ARRAY)});
  task.add_scalar_arg(legate::Scalar{false});
  runtime->submit(std::move(task));

  // Check task
  task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::CHECK_ARRAY_TASK_ID));
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::PRIMITIVE_ARRAY)});
  task.add_scalar_arg(legate::Scalar{false});
  runtime->submit(std::move(task));
}

void test_fill_unbound_primitive_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto dim           = 2;
  auto logical_array = runtime->create_array(legate::int32(), dim, nullable);

  // Fill task
  auto task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::FILL_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::PRIMITIVE_ARRAY)});
  task.add_scalar_arg(legate::Scalar{true});
  runtime->submit(std::move(task));

  // Check task
  task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::CHECK_ARRAY_TASK_ID));
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::PRIMITIVE_ARRAY)});
  task.add_scalar_arg(legate::Scalar{true});
  runtime->submit(std::move(task));
}

void test_fill_bound_list_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto arr_type      = legate::list_type(legate::int64()).as_list_type();
  auto logical_array = runtime->create_array({3}, arr_type, nullable);

  // Fill task
  auto task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::FILL_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::LIST_ARRAY)});
  task.add_scalar_arg(legate::Scalar{false});
  runtime->submit(std::move(task));

  // Check task
  task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::CHECK_ARRAY_TASK_ID));
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::LIST_ARRAY)});
  task.add_scalar_arg(legate::Scalar{false});
  runtime->submit(std::move(task));
}

void test_fill_unbound_list_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto arr_type      = legate::list_type(legate::int64()).as_list_type();
  auto dim           = 1;
  auto logical_array = runtime->create_array(arr_type, dim, nullable);

  // Fill task
  auto task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::FILL_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::LIST_ARRAY)});
  task.add_scalar_arg(legate::Scalar{true});
  runtime->submit(std::move(task));

  // Check task
  task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::CHECK_ARRAY_TASK_ID));
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::LIST_ARRAY)});
  task.add_scalar_arg(legate::Scalar{true});
  runtime->submit(std::move(task));
}

void test_fill_bound_string_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto arr_type      = legate::string_type();
  auto logical_array = runtime->create_array({3}, arr_type, nullable);

  // Fill task
  auto task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::FILL_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::STRING_TYPE)});
  task.add_scalar_arg(legate::Scalar{false});
  runtime->submit(std::move(task));

  // Check task
  task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::CHECK_ARRAY_TASK_ID));
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::STRING_TYPE)});
  task.add_scalar_arg(legate::Scalar{false});
  runtime->submit(std::move(task));
}

void test_fill_unbound_string_array(bool nullable)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(library_name);
  auto arr_type      = legate::string_type();
  auto dim           = 1;
  auto logical_array = runtime->create_array(arr_type, dim, nullable);

  // Fill task
  auto task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::FILL_ARRAY_TASK_ID));
  auto part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::STRING_TYPE)});
  task.add_scalar_arg(legate::Scalar{true});
  runtime->submit(std::move(task));

  // Check task
  task = runtime->create_task(context, static_cast<int64_t>(ArrayTaskID::CHECK_ARRAY_TASK_ID));
  part = task.declare_partition();
  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(
    legate::Scalar{static_cast<std::underlying_type_t<ArrayType>>(ArrayType::STRING_TYPE)});
  task.add_scalar_arg(legate::Scalar{true});
  runtime->submit(std::move(task));
}

void test_fill_primitive(bool nullable)
{
  register_tasks();
  test_fill_bound_primitive_array(nullable);
  test_fill_unbound_primitive_array(nullable);
}

void test_fill_list(bool nullable)
{
  register_tasks();
  test_fill_bound_list_array(nullable);
  test_fill_unbound_list_array(nullable);
}

void test_fill_string(bool nullable)
{
  register_tasks();
  test_fill_bound_string_array(nullable);
  test_fill_unbound_string_array(nullable);
}

TEST_F(PhysicalArrayUnit, CreatePrimitiveNonNullable) { test_primitive_array(false); }

TEST_F(PhysicalArrayUnit, CreatePrimitiveNullable) { test_primitive_array(true); }

TEST_F(PhysicalArrayUnit, CreateListNonNullable) { test_list_array(false); }

TEST_F(PhysicalArrayUnit, CreateListNullable) { test_list_array(true); }

TEST_F(PhysicalArrayUnit, CreateStringNonNullable) { test_string_array(false); }

TEST_F(PhysicalArrayUnit, CreateStringNullable) { test_string_array(true); }

TEST_F(PhysicalArrayUnit, FillPrimitiveNonNullable) { test_fill_primitive(false); }

// TODO(issue 374)
// TEST_F(PhysicalArrayUnit, FillPrimitiveNullable) { test_fill_primitive(true); }

TEST_F(PhysicalArrayUnit, FillListNonNullable) { test_fill_list(false); }

// TODO(issue 374)
// TEST_F(PhysicalArrayUnit, FillListNullable) { test_fill_list(true); }

TEST_F(PhysicalArrayUnit, FillStringNonNullable) { test_fill_string(false); }

// TODO(issue 374)
// TEST_F(PhysicalArrayUnit, FillStringNullable) { test_fill_string(true); }

}  // namespace physical_array_unit_test
