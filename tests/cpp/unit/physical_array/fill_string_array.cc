/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_array_fill_string_test {

namespace {

constexpr std::int64_t SIZE_ARRAY = 10;

class FillStringTask : public legate::LegateTask<FillStringTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class CheckStringTask : public legate::LegateTask<CheckStringTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_fill_string_physical_array";

  static void registration_callback(legate::Library library)
  {
    FillStringTask::register_variants(library);
    CheckStringTask::register_variants(library);
  }
};

class FillStringPhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class NullableFillStringArrayTest : public FillStringPhysicalArrayUnit,
                                    public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(FillStringPhysicalArrayUnit,
                         NullableFillStringArrayTest,
                         ::testing::Values(true, false));

template <typename ACC>
void fill_string_array(const ACC& desc,
                       const legate::Buffer<char, 1>& chars,
                       const legate::Rect<1>& desc_shape,
                       std::int64_t chars_size)
{
  // Here we create a list of the following form:
  // idx       0         1             2                9
  //       +-------+-----------+---------------+- ... -------+
  //       | +---+ | +---+---+ | +---+---+---+ |  ... -+---+ |
  // val   | | a | | | a | b | | | a | b | c | |  ...  | j | |
  //       | +---+ | +---+---+ | +---+---+---+ |  ... -+---+ |
  //       +-------+-----------+---------------+- ... -------+
  std::int64_t chars_offset = 0;

  for (legate::PointInRectIterator<1> it{desc_shape}; it.valid(); ++it) {
    const auto range = legate::Rect<1>{chars_offset, chars_offset + (*it)[0]};

    desc[*it] = range;
    for (legate::PointInRectIterator<1> vit{range}; vit.valid(); ++vit) {
      chars[*vit] = static_cast<char>((*vit - range.lo)[0] + 'a');
    }
    chars_offset += static_cast<std::int64_t>(range.volume());
  }

  ASSERT_EQ(chars_size, chars_offset);
}

template <typename ACC>
void fill_null_mask(const ACC& acc, const legate::Rect<1>& shape)
{
  for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
    acc[*it] = (*it)[0] % 2 == 0;
  }
}

/*static*/ void FillStringTask::cpu_variant(legate::TaskContext context)
{
  auto string_array = context.output(0).as_string_array();
  auto ranges       = string_array.ranges().data();
  auto chars_store  = string_array.chars().data();
  auto unbound      = context.scalar(1).value<bool>();

  constexpr std::int64_t SIZE_CHARS = ((1 + SIZE_ARRAY) * SIZE_ARRAY) / 2;

  const auto chars = chars_store.create_output_buffer<char, 1>(legate::Point<1>{SIZE_CHARS}, true);

  if (unbound) {
    auto desc = ranges.create_output_buffer<legate::Rect<1>, 1>(legate::Point<1>{SIZE_ARRAY}, true);

    fill_string_array(desc, chars, legate::Rect<1>{0, SIZE_ARRAY - 1}, SIZE_CHARS);
  } else {
    fill_string_array(
      ranges.write_accessor<legate::Rect<1>, 1>(), chars, ranges.shape<1>(), SIZE_CHARS);
  }

  const auto nullable = context.scalar(0).value<bool>();

  if (nullable) {
    auto null_mask = string_array.null_mask();

    if (null_mask.is_unbound_store()) {
      auto mask = null_mask.create_output_buffer<bool, 1>(legate::Point<1>{SIZE_ARRAY}, true);

      fill_null_mask(mask, legate::Rect<1>{0, SIZE_ARRAY - 1});
    } else {
      fill_null_mask(null_mask.write_accessor<bool, 1>(), null_mask.shape<1>());
    }
  }
}

/*static*/ void CheckStringTask::cpu_variant(legate::TaskContext context)
{
  auto string_array = context.input(0).as_string_array();
  const auto shape  = string_array.shape<1>();
  auto ranges       = string_array.ranges().data().read_accessor<legate::Rect<1>, 1>();
  auto chars        = string_array.chars().data().read_accessor<char, 1>();

  for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
    const auto& range = ranges[*it];

    for (legate::PointInRectIterator<1> vit{range}; vit.valid(); ++vit) {
      ASSERT_EQ(chars[*vit], static_cast<char>((*vit - range.lo)[0] + 'a'));
    }
  }

  const auto nullable = context.scalar(0).value<bool>();

  if (nullable) {
    auto null_mask = string_array.null_mask().read_accessor<bool, 1>();

    for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
      ASSERT_EQ(null_mask[*it], (*it)[0] % 2 == 0);
    }
  }
}

void test_fill_string_array_task(const legate::LogicalArray& logical_array,
                                 bool nullable,
                                 bool unbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, FillStringTask::TASK_CONFIG.task_id());

  auto part = task.add_output(logical_array);
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  if (!unbound) {
    task.add_constraint(legate::broadcast(part));
  }
  runtime->submit(std::move(task));

  task = runtime->create_task(context, CheckStringTask::TASK_CONFIG.task_id());
  task.add_input(logical_array);
  task.add_scalar_arg(legate::Scalar{nullable});
  task.add_scalar_arg(legate::Scalar{unbound});
  runtime->submit(std::move(task));
}

}  // namespace

TEST_P(NullableFillStringArrayTest, BoundStringArray)
{
  const auto nullable = GetParam();
  auto logical_array =
    legate::Runtime::get_runtime()->create_array({SIZE_ARRAY}, legate::string_type(), nullable);

  test_fill_string_array_task(logical_array, nullable, false);
}

TEST_P(NullableFillStringArrayTest, UnboundStringArray)
{
  const auto nullable = GetParam();
  auto logical_array =
    legate::Runtime::get_runtime()->create_array(legate::string_type(), 1, nullable);

  test_fill_string_array_task(logical_array, nullable, true);
}

}  // namespace physical_array_fill_string_test
