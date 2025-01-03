/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/utilities/detail/zip.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <array>
#include <fmt/format.h>
#include <gtest/gtest.h>

namespace register_variants {

// NOLINTBEGIN(readability-magic-numbers)

using RegisterVariants = DefaultFixture;

namespace {

enum TaskID : std::uint8_t {
  HELLO1,
  HELLO2,
  HELLO3,
  HELLO4,
  HELLO5,
  HELLO6,
};

constexpr std::string_view LIBRARY_NAME1 = "test_register_variants1";

struct Registry {
  [[nodiscard]] static legate::TaskRegistrar& get_registrar();
};

legate::TaskRegistrar& Registry::get_registrar()
{
  static legate::TaskRegistrar registrar{};

  return registrar;
}

void hello_cpu_variant(legate::TaskContext& context)
{
  auto output = context.output(0).data();
  auto shape  = output.shape<2>();

  if (shape.empty()) {
    return;
  }

  auto acc = output.write_accessor<std::int64_t, 2>(shape);
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    acc[*it] = (*it)[0] + (*it)[1] * 1000;
  }
}

}  // namespace

// Do not put these in the anon namespace, we have a test which checks their name, and anon
// namespaces have implementation-defined names.
template <std::int32_t TID>
struct BaseTask : public legate::LegateTask<BaseTask<TID>> {
  using Registrar               = Registry;
  static constexpr auto TASK_ID = legate::LocalTaskID{TID};
  static void cpu_variant(legate::TaskContext context) { hello_cpu_variant(context); }
};

struct BaseTask2 : public legate::LegateTask<BaseTask2> {
  static void cpu_variant(legate::TaskContext context) { hello_cpu_variant(context); }
};

namespace {

void test_auto_task(const legate::Library& context,
                    const legate::LogicalStore& store,
                    TaskID taskid)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, legate::LocalTaskID{taskid});
  auto part    = task.declare_partition();
  task.add_output(store, part);
  runtime->submit(std::move(task));
}

void test_manual_task(const legate::Library& context,
                      const legate::LogicalStore& store,
                      TaskID taskid)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task =
    runtime->create_task(context, legate::LocalTaskID{taskid}, legate::tuple<std::uint64_t>{3, 3});
  auto part = store.partition_by_tiling({2, 2});
  task.add_output(part);
  runtime->submit(std::move(task));
}

void validate_store(const legate::LogicalStore& store)
{
  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<std::int64_t, 2>();
  auto shape   = p_store.shape<2>();
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    EXPECT_EQ(acc[*it], (*it)[0] + ((*it)[1] * 1000));
  }
}

void verify_test(const legate::Library& context, TaskID taskid)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{5, 5}, legate::int64());
  test_auto_task(context, store, taskid);
  validate_store(store);

  test_manual_task(context, store, taskid);
  validate_store(store);
}

}  // namespace

TEST_F(RegisterVariants, Test1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_or_create_library(LIBRARY_NAME1);

  EXPECT_THROW(static_cast<void>(context.get_task_name(legate::LocalTaskID{HELLO1})),
               std::out_of_range);

  using HelloTask = BaseTask<HELLO1>;
  HelloTask::register_variants();
  Registry::get_registrar().register_all_tasks(context);

  // Sanity test that tasks are registered successfully
  const std::string task_name =
    fmt::format("register_variants::BaseTask<{}>", fmt::underlying(HELLO1));
  EXPECT_EQ(context.get_task_name(legate::LocalTaskID{HELLO1}), task_name);

  verify_test(context, HELLO1);
}

TEST_F(RegisterVariants, Test2)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_or_create_library("test_register_variants1");

  EXPECT_THROW(static_cast<void>(context.get_task_name(legate::LocalTaskID{HELLO2})),
               std::out_of_range);

  std::map<legate::VariantCode, legate::VariantOptions> all_options;
  all_options[legate::VariantCode::CPU] = legate::VariantOptions{};
  all_options[legate::VariantCode::GPU] = legate::VariantOptions{};

  using HelloTask = BaseTask<HELLO2>;
  HelloTask::register_variants(all_options);
  Registry::get_registrar().register_all_tasks(context);

  // Sanity test that tasks are registered successfully
  const std::string task_name =
    fmt::format("register_variants::BaseTask<{}>", fmt::underlying(HELLO2));
  EXPECT_EQ(context.get_task_name(legate::LocalTaskID{HELLO2}), task_name);

  verify_test(context, HELLO2);
}

TEST_F(RegisterVariants, Test3)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_or_create_library("test_register_variants1");

  EXPECT_THROW(static_cast<void>(context.get_task_name(legate::LocalTaskID{HELLO3})),
               std::out_of_range);

  using HelloTask = BaseTask<HELLO3>;
  HelloTask::register_variants(context);

  // registered taskID the second time would throw exception
  EXPECT_THROW(HelloTask::register_variants(context), std::invalid_argument);
  // Sanity test that tasks are registered successfully
  const std::string task_name =
    fmt::format("register_variants::BaseTask<{}>", fmt::underlying(HELLO3));
  EXPECT_EQ(context.get_task_name(legate::LocalTaskID{HELLO3}), task_name);

  verify_test(context, HELLO3);
}

TEST_F(RegisterVariants, Test4)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_or_create_library("test_register_variants1");

  EXPECT_THROW(static_cast<void>(context.get_task_name(legate::LocalTaskID{HELLO4})),
               std::out_of_range);

  std::map<legate::VariantCode, legate::VariantOptions> all_options;
  all_options[legate::VariantCode::CPU] = legate::VariantOptions{};
  all_options[legate::VariantCode::GPU] = legate::VariantOptions{};

  using HelloTask = BaseTask<HELLO4>;
  HelloTask::register_variants(context, all_options);

  // registered taskID the second time would throw exception
  EXPECT_THROW(HelloTask::register_variants(context), std::invalid_argument);
  // Sanity test that tasks are registered successfully
  const std::string task_name =
    fmt::format("register_variants::BaseTask<{}>", fmt::underlying(HELLO4));
  EXPECT_EQ(context.get_task_name(legate::LocalTaskID{HELLO4}), task_name);

  verify_test(context, HELLO4);
}

TEST_F(RegisterVariants, Test5)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_or_create_library("test_register_variants1");

  EXPECT_THROW(static_cast<void>(context.get_task_name(legate::LocalTaskID{HELLO5})),
               std::out_of_range);

  BaseTask2::register_variants(context, legate::LocalTaskID{HELLO5});

  // registered taskID the second time would throw exception
  EXPECT_THROW(BaseTask2::register_variants(context, legate::LocalTaskID{HELLO5}),
               std::invalid_argument);
  // Sanity test that tasks are registered successfully
  EXPECT_EQ(context.get_task_name(legate::LocalTaskID{HELLO5}), "register_variants::BaseTask2");

  verify_test(context, HELLO5);
}

TEST_F(RegisterVariants, Test6)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_or_create_library("test_register_variants1");

  EXPECT_THROW(static_cast<void>(context.get_task_name(legate::LocalTaskID{HELLO6})),
               std::out_of_range);

  std::map<legate::VariantCode, legate::VariantOptions> all_options;
  all_options[legate::VariantCode::CPU] = legate::VariantOptions{};
  all_options[legate::VariantCode::GPU] = legate::VariantOptions{};

  BaseTask2::register_variants(context, legate::LocalTaskID{HELLO6}, all_options);

  // registered taskID the second time would throw exception
  EXPECT_THROW(BaseTask2::register_variants(context, legate::LocalTaskID{HELLO6}, all_options),
               std::invalid_argument);
  // Sanity test that tasks are registered successfully
  EXPECT_EQ(context.get_task_name(legate::LocalTaskID{HELLO6}), "register_variants::BaseTask2");

  verify_test(context, HELLO6);
}

class DefaultOptionsTask : public legate::LegateTask<DefaultOptionsTask> {
 public:
  static constexpr auto TASK_ID             = legate::LocalTaskID{0};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);
  static constexpr auto OMP_VARIANT_OPTIONS = legate::VariantOptions{}.with_return_size(4567);
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_return_size(1234);

  static void cpu_variant(legate::TaskContext) {}
  static void omp_variant(legate::TaskContext) {}
  static void gpu_variant(legate::TaskContext) {}
};

TEST_F(RegisterVariants, DefaultVariantOptions)
{
  auto library = legate::Runtime::get_runtime()->create_library("test_register_variants2");

  DefaultOptionsTask::register_variants(library);

  const auto* task_info = library.find_task(DefaultOptionsTask::TASK_ID);
  // This test checks that the defaults in <XXX>_VARIANT_OPTIONS override the "normal"
  // defaults. Obviously, we cannot properly test that if the normal defaults match that of
  // DefaultOptionsTask::<XXX>_VARIANT_OPTIONS.
  static_assert(legate::VariantOptions::DEFAULT_OPTIONS != DefaultOptionsTask::CPU_VARIANT_OPTIONS);
  static_assert(legate::VariantOptions::DEFAULT_OPTIONS != DefaultOptionsTask::OMP_VARIANT_OPTIONS);
  static_assert(legate::VariantOptions::DEFAULT_OPTIONS != DefaultOptionsTask::GPU_VARIANT_OPTIONS);

  constexpr std::array variant_kinds = {
    legate::VariantCode::CPU, legate::VariantCode::OMP, legate::VariantCode::GPU};
  constexpr std::array options = {DefaultOptionsTask::CPU_VARIANT_OPTIONS,
                                  DefaultOptionsTask::OMP_VARIANT_OPTIONS,
                                  DefaultOptionsTask::GPU_VARIANT_OPTIONS};

  for (auto&& [variant_kind, default_options] : legate::detail::zip_equal(variant_kinds, options)) {
    const auto variant = task_info->find_variant(variant_kind);

    ASSERT_TRUE(variant.has_value());
    // We do check it, immediately above! But for some reason clang-tidy doesn't clock that...
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    ASSERT_EQ(variant->get().options, default_options);
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace register_variants
