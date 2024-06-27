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

#include "core/utilities/detail/zip.h"

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
  HELLO  = 0,
  HELLO1 = 1,
  HELLO2 = 2,
  HELLO3 = 3,
  HELLO4 = 4,
  HELLO5 = 5,
};

constexpr std::array<TaskID, 6> TASK_IDS = {HELLO, HELLO1, HELLO2, HELLO3, HELLO4, HELLO5};

}  // namespace

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

  auto acc = output.write_accessor<int64_t, 2>(shape);
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    acc[*it] = (*it)[0] + (*it)[1] * 1000;
  }
}

template <std::int32_t TID>
struct BaseTask : public legate::LegateTask<BaseTask<TID>> {
  using Registrar                       = Registry;
  static constexpr std::int32_t TASK_ID = TID;
  static void cpu_variant(legate::TaskContext context) { hello_cpu_variant(context); }
};

struct BaseTask2 : public legate::LegateTask<BaseTask2> {
  static void cpu_variant(legate::TaskContext context) { hello_cpu_variant(context); }
};

void test_register_tasks(const legate::Library& context)
{
  using HelloTask  = BaseTask<HELLO>;
  using HelloTask1 = BaseTask<HELLO1>;
  using HelloTask2 = BaseTask<HELLO2>;
  using HelloTask3 = BaseTask<HELLO3>;

  std::map<legate::LegateVariantCode, legate::VariantOptions> all_options;
  all_options[LEGATE_CPU_VARIANT] = legate::VariantOptions{};
  all_options[LEGATE_GPU_VARIANT] = legate::VariantOptions{};

  {
    HelloTask::register_variants();
    HelloTask1::register_variants(all_options);
    Registry::get_registrar().register_all_tasks(context);
  }

  HelloTask2::register_variants(context);

  HelloTask3::register_variants(context, all_options);

  BaseTask2::register_variants(context, HELLO4);
  BaseTask2::register_variants(context, HELLO5, all_options);

  // registered taskID the second time would throw exception
  EXPECT_THROW(HelloTask2::register_variants(context), std::invalid_argument);
  EXPECT_THROW(BaseTask2::register_variants(context, HELLO4, all_options), std::invalid_argument);
  EXPECT_THROW(BaseTask2::register_variants(context, HELLO2), std::invalid_argument);
}

void test_auto_task(const legate::Library& context,
                    const legate::LogicalStore& store,
                    TaskID taskid)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, taskid);
  auto part    = task.declare_partition();
  task.add_output(store, part);
  runtime->submit(std::move(task));
}

void test_manual_task(const legate::Library& context,
                      const legate::LogicalStore& store,
                      TaskID taskid)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, taskid, legate::tuple<std::uint64_t>{3, 3});
  auto part    = store.partition_by_tiling({2, 2});
  task.add_output(part);
  runtime->submit(std::move(task));
}

void validate_store(const legate::LogicalStore& store)
{
  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<int64_t, 2>();
  auto shape   = p_store.shape<2>();
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    EXPECT_EQ(acc[*it], (*it)[0] + (*it)[1] * 1000);
  }
}

TEST_F(RegisterVariants, All)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library("test_register_variants1");

  for (auto task_id : TASK_IDS) {
    EXPECT_THROW(static_cast<void>(context.get_task_name(task_id)), std::out_of_range);
  }

  test_register_tasks(context);

  // Sanity test that tasks are registered successfully
  for (auto task_id : TASK_IDS) {
    const std::string task_name =
      task_id >= HELLO4 ? "register_variants::BaseTask2"
                        : fmt::format("register_variants::BaseTask<{}>", fmt::underlying(task_id));
    EXPECT_STREQ(context.get_task_name(task_id).data(), task_name.c_str());
  }

  auto store = runtime->create_store(legate::Shape{5, 5}, legate::int64());
  for (auto task_id : TASK_IDS) {
    test_auto_task(context, store, task_id);
    validate_store(store);
  }

  for (auto task_id : TASK_IDS) {
    test_manual_task(context, store, task_id);
    validate_store(store);
  }
}

class DefaultOptionsTask : public legate::LegateTask<DefaultOptionsTask> {
 public:
  static constexpr std::int32_t TASK_ID     = 0;
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);
  static constexpr auto OMP_VARIANT_OPTIONS = legate::VariantOptions{}.with_leaf(false);
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

  constexpr std::array variant_kinds = {LEGATE_CPU_VARIANT, LEGATE_OMP_VARIANT, LEGATE_GPU_VARIANT};
  constexpr std::array options       = {DefaultOptionsTask::CPU_VARIANT_OPTIONS,
                                        DefaultOptionsTask::OMP_VARIANT_OPTIONS,
                                        DefaultOptionsTask::GPU_VARIANT_OPTIONS};

  for (auto&& [variant_kind, default_options] : legate::detail::zip(variant_kinds, options)) {
    ASSERT_TRUE(task_info->has_variant(variant_kind));
    ASSERT_EQ(task_info->find_variant(variant_kind).options, default_options);
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace register_variants
