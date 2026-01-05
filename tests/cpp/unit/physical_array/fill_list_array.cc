/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_array_fill_list_test {

namespace {

constexpr std::int64_t SIZE_ARRAY = 10;

class FillListTask : public legate::LegateTask<FillListTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class CheckListTask : public legate::LegateTask<CheckListTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_fill_list_physical_array";

  static void registration_callback(legate::Library library)
  {
    FillListTask::register_variants(library);
    CheckListTask::register_variants(library);
  }
};

class FillListPhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class NullableFillListArrayTest : public FillListPhysicalArrayUnit,
                                  public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(FillListPhysicalArrayUnit,
                         NullableFillListArrayTest,
                         ::testing::Values(true, false));

template <typename ACC>
void fill_list_array(const ACC& desc,
                     const legate::Buffer<std::int64_t, 1>& vardata,
                     const legate::Rect<1>& desc_shape,
                     std::int64_t vardata_size)
{
  // Here we create a list of the following form:
  // idx       0         1             2                9
  //       +-------+-----------+---------------+- ... -------+
  //       | +---+ | +---+---+ | +---+---+---+ |  ... -+---+ |
  // val   | | 0 | | | 0 | 1 | | | 0 | 1 | 2 | |  ...  | 9 | |
  //       | +---+ | +---+---+ | +---+---+---+ |  ... -+---+ |
  //       +-------+-----------+---------------+- ... -------+
  std::int64_t vardata_offset = 0;

  for (legate::PointInRectIterator<1> it{desc_shape}; it.valid(); ++it) {
    const auto range = legate::Rect<1>{vardata_offset, vardata_offset + (*it)[0]};

    desc[*it] = range;
    for (legate::PointInRectIterator<1> vit{range}; vit.valid(); ++vit) {
      vardata[*vit] = (*vit - range.lo)[0];
    }
    vardata_offset += static_cast<std::int64_t>(range.volume());
  }

  ASSERT_EQ(vardata_size, vardata_offset);
}

template <typename ACC>
void fill_null_mask(const ACC& acc, const legate::Rect<1>& shape)
{
  for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
    acc[*it] = (*it)[0] % 2 == 0;
  }
}

/*static*/ void FillListTask::cpu_variant(legate::TaskContext context)
{
  auto list_array    = context.output(0).as_list_array();
  auto descriptor    = list_array.descriptor().data();
  auto vardata_store = list_array.vardata().data();
  auto unbound       = context.scalar(1).value<bool>();

  constexpr std::int64_t SIZE_VARDATA = ((1 + SIZE_ARRAY) * SIZE_ARRAY) / 2;

  const auto vardata = vardata_store.create_output_buffer<std::int64_t, 1>(
    legate::Point<1>{SIZE_VARDATA}, /*bind_buffer=*/true);

  if (unbound) {
    auto desc = descriptor.create_output_buffer<legate::Rect<1>, 1>(legate::Point<1>{SIZE_ARRAY},
                                                                    /*bind_buffer=*/true);

    fill_list_array(desc, vardata, legate::Rect<1>{0, SIZE_ARRAY - 1}, SIZE_VARDATA);
  } else {
    fill_list_array(descriptor.write_accessor<legate::Rect<1>, 1>(),
                    vardata,
                    descriptor.shape<1>(),
                    SIZE_VARDATA);
  }

  const auto nullable = context.scalar(0).value<bool>();

  if (nullable) {
    auto null_mask = list_array.null_mask();

    if (null_mask.is_unbound_store()) {
      auto mask =
        null_mask.create_output_buffer<bool, 1>(legate::Point<1>{SIZE_ARRAY}, /*bind_buffer=*/true);

      fill_null_mask(mask, legate::Rect<1>{0, SIZE_ARRAY - 1});
    } else {
      fill_null_mask(null_mask.write_accessor<bool, 1>(), null_mask.shape<1>());
    }
  }
}

/*static*/ void CheckListTask::cpu_variant(legate::TaskContext context)
{
  auto list_array  = context.input(0).as_list_array();
  const auto shape = list_array.shape<1>();
  auto descriptor  = list_array.descriptor().data().read_accessor<legate::Rect<1>, 1>();
  auto vardata     = list_array.vardata().data().read_accessor<std::int64_t, 1>();

  for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
    const auto& range = descriptor[*it];

    for (legate::PointInRectIterator<1> vit{range}; vit.valid(); ++vit) {
      ASSERT_EQ(vardata[*vit], (*vit - range.lo)[0]);
    }
  }

  const auto nullable = context.scalar(0).value<bool>();

  if (nullable) {
    auto null_mask = list_array.null_mask().read_accessor<bool, 1>();

    for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
      ASSERT_EQ(null_mask[*it], (*it)[0] % 2 == 0);
    }
  }
}

void test_fill_list_array_task(const legate::LogicalArray& logical_array,
                               bool nullable,
                               bool unbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, FillListTask::TASK_CONFIG.task_id());

  auto part = task.add_output(logical_array);
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  if (!unbound) {
    task.add_constraint(legate::broadcast(part));
  }
  runtime->submit(std::move(task));

  task = runtime->create_task(context, CheckListTask::TASK_CONFIG.task_id());
  task.add_input(logical_array);
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(NullableFillListArrayTest, BoundListArray)
{
  const auto nullable = GetParam();
  auto logical_array  = legate::Runtime::get_runtime()->create_array(
    {SIZE_ARRAY}, legate::list_type(legate::int64()), nullable);

  test_fill_list_array_task(logical_array, nullable, /*unbound=*/false);
}

TEST_P(NullableFillListArrayTest, UnboundListArray)
{
  const auto nullable = GetParam();
  auto logical_array  = legate::Runtime::get_runtime()->create_array(
    legate::list_type(legate::int64()), /*dim=*/1, nullable);

  test_fill_list_array_task(logical_array, nullable, /*unbound=*/true);
}

}  // namespace physical_array_fill_list_test
