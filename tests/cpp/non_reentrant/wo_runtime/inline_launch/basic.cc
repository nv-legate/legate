/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_inline_launch_basic {

namespace {

constexpr auto MANUAL_TASK_OUTPUT_VALUE      = std::int64_t{42};
constexpr auto PHYSICAL_TASK_INPUT_VALUE     = std::int64_t{7};
constexpr auto PHYSICAL_TASK_OUTPUT_VALUE    = std::int64_t{13};
constexpr auto PHYSICAL_TASK_REDUCTION_VALUE = std::int64_t{5};
constexpr std::string_view INLINE_TASK_PROVENANCE{"inline-physical-task-provenance"};

class CheckTask : public legate::LegateTask<CheckTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(legate::TaskSignature{}.inputs(1));

  static void cpu_variant(legate::TaskContext context)
  {
    const auto arg = context.input(0);

    ASSERT_EQ(arg.target(), legate::mapping::StoreTarget::SYSMEM);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    const auto arg = context.input(0);

    ASSERT_EQ(arg.target(), legate::mapping::StoreTarget::FBMEM);
  }

  static void omp_variant(legate::TaskContext context)
  {
    const auto arg = context.input(0);

    ASSERT_EQ(arg.target(), legate::mapping::StoreTarget::SOCKETMEM);
  }
};

class ManualCheckTask : public legate::LegateTask<ManualCheckTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1));

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_TRUE(context.is_single_task());
    ASSERT_EQ(context.get_launch_domain().get_volume(), 1);
    ASSERT_EQ(context.num_inputs(), 1);
    ASSERT_EQ(context.num_outputs(), 1);

    const auto input = context.input(0);
    auto output      = context.output(0);
    auto output_span = output.span_write_accessor<std::int64_t, 1>();

    ASSERT_EQ(input.target(), legate::mapping::StoreTarget::SYSMEM);
    ASSERT_EQ(output.target(), legate::mapping::StoreTarget::SYSMEM);
    ASSERT_EQ(output_span.extent(0), std::size_t{1});

    output_span[0] = MANUAL_TASK_OUTPUT_VALUE;
  }
};

class PhysicalStoreTask : public legate::LegateTask<PhysicalStoreTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).redops(1));

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_TRUE(context.is_single_task());
    ASSERT_EQ(context.num_inputs(), 1);
    ASSERT_EQ(context.num_outputs(), 1);
    ASSERT_EQ(context.num_reductions(), 1);

    const auto input     = context.input(0);
    auto output          = context.output(0);
    auto reduction       = context.reduction(0);
    auto input_span      = input.span_read_accessor<std::int64_t, 1>();
    auto output_span     = output.span_write_accessor<std::int64_t, 1>();
    const auto red_acc   = reduction.reduce_accessor<Legion::SumReduction<std::int64_t>, true, 1>();
    const auto red_shape = reduction.shape<1>();

    ASSERT_EQ(input.target(), legate::mapping::StoreTarget::SYSMEM);
    ASSERT_EQ(output.target(), legate::mapping::StoreTarget::SYSMEM);
    ASSERT_EQ(reduction.target(), legate::mapping::StoreTarget::SYSMEM);
    ASSERT_EQ(input_span.extent(0), std::size_t{1});
    ASSERT_EQ(output_span.extent(0), std::size_t{1});
    ASSERT_EQ(red_shape.volume(), 1);
    ASSERT_EQ(input_span[0], PHYSICAL_TASK_INPUT_VALUE);

    output_span[0] = PHYSICAL_TASK_OUTPUT_VALUE;
    red_acc.reduce(legate::Point<1>{0}, PHYSICAL_TASK_REDUCTION_VALUE);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_inline_launch_basic";

  static void registration_callback(legate::Library library)
  {
    CheckTask::register_variants(library);
    ManualCheckTask::register_variants(library);
    PhysicalStoreTask::register_variants(library);
  }
};

class InlineLaunchUnit : public RegisterOnceFixture<Config> {
 protected:
  void SetUp() override
  {
    ASSERT_NO_THROW(legate::start());
    RegisterOnceFixture::SetUp();
  }

  void TearDown() override
  {
    RegisterOnceFixture::TearDown();
    ASSERT_EQ(legate::finish(), 0);
  }

 private:
  legate::test::Environment::TemporaryEnvVar legate_config_{"LEGATE_CONFIG",
                                                            /*value=*/"--inline-task-launch ",
                                                            /* overwrite */ true};
};

[[nodiscard]] legate::LogicalStore make_store(const legate::Shape& shape, std::int64_t value = 0)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto ret            = runtime->create_store(shape, legate::int64());

  runtime->issue_fill(ret, legate::Scalar{value});
  return ret;
}

[[nodiscard]] legate::AutoTask create_physical_store_task()
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);

  return runtime->create_task(lib, PhysicalStoreTask::TASK_CONFIG.task_id());
}

void assert_physical_task_result(const legate::LogicalStore& output,
                                 const legate::LogicalStore& reduction)
{
  const auto output_span    = output.get_physical_store().span_read_accessor<std::int64_t, 1>();
  const auto reduction_span = reduction.get_physical_store().span_read_accessor<std::int64_t, 1>();

  ASSERT_EQ(output_span.extent(0), std::size_t{1});
  ASSERT_EQ(reduction_span.extent(0), std::size_t{1});
  ASSERT_EQ(output_span[0], PHYSICAL_TASK_OUTPUT_VALUE);
  ASSERT_EQ(reduction_span[0], PHYSICAL_TASK_REDUCTION_VALUE);
}

class StoreTarget : public InlineLaunchUnit,
                    public ::testing::WithParamInterface<legate::mapping::TaskTarget> {};

}  // namespace

INSTANTIATE_TEST_SUITE_P(InlineLaunchUnit,
                         StoreTarget,
                         ::testing::Values(legate::mapping::TaskTarget::CPU,
                                           legate::mapping::TaskTarget::GPU,
                                           legate::mapping::TaskTarget::OMP));

TEST_P(StoreTarget, Basic)
{
  const auto target   = GetParam();
  auto* const runtime = legate::Runtime::get_runtime();
  const auto machine  = runtime->get_machine().only(target);

  if (machine.empty()) {
    GTEST_SKIP() << "Test requires " << target;
  }

  const auto _ = legate::Scope{machine};

  const auto lib   = runtime->find_library(Config::LIBRARY_NAME);
  auto task        = runtime->create_task(lib, CheckTask::TASK_CONFIG.task_id());
  const auto store = make_store(legate::Shape{5, 1});

  task.add_input(store);
  runtime->submit(std::move(task));
}

TEST_F(InlineLaunchUnit, ManualTask)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto machine  = runtime->get_machine().only(legate::mapping::TaskTarget::CPU);

  if (machine.empty()) {
    GTEST_SKIP() << "Test requires CPU";
  }

  const auto _ = legate::Scope{machine};

  const auto lib   = runtime->find_library(Config::LIBRARY_NAME);
  auto task        = runtime->create_task(lib, ManualCheckTask::TASK_CONFIG.task_id(), {1});
  const auto input = make_store(legate::Shape{1});
  auto output      = runtime->create_store(legate::Shape{1}, legate::int64());

  task.add_input(input);
  task.add_output(output);
  runtime->submit(std::move(task));

  const auto output_span = output.get_physical_store().span_read_accessor<std::int64_t, 1>();

  ASSERT_EQ(output_span.extent(0), std::size_t{1});
  ASSERT_EQ(output_span[0], MANUAL_TASK_OUTPUT_VALUE);
}

TEST_F(InlineLaunchUnit, PhysicalTaskReduction)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto machine  = runtime->get_machine().only(legate::mapping::TaskTarget::CPU);

  if (machine.empty()) {
    GTEST_SKIP() << "Test requires CPU";
  }

  const auto _ = legate::Scope{machine};

  auto task                    = create_physical_store_task();
  const auto shape             = legate::Shape{1};
  const auto input             = make_store(shape, PHYSICAL_TASK_INPUT_VALUE);
  const auto output            = runtime->create_store(shape, legate::int64());
  const auto reduction         = make_store(shape);
  const auto redop_kind        = legate::ReductionOpKind::ADD;
  const auto legion_redop_kind = static_cast<std::int32_t>(redop_kind);

  task.add_input(input);
  task.add_output(output);
  ASSERT_EQ(task.add_reduction(reduction, legion_redop_kind).impl(), nullptr);
  runtime->submit(std::move(task));

  assert_physical_task_result(output, reduction);
}

TEST_F(InlineLaunchUnit, InlinePhysicalTaskIgnoresPartitionSymbols)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto machine  = runtime->get_machine().only(legate::mapping::TaskTarget::CPU);

  if (machine.empty()) {
    GTEST_SKIP() << "Test requires CPU";
  }

  const auto _ = legate::Scope{machine};

  auto task                    = create_physical_store_task();
  const auto shape             = legate::Shape{1};
  const auto input             = make_store(shape, PHYSICAL_TASK_INPUT_VALUE);
  const auto output            = runtime->create_store(shape, legate::int64());
  const auto reduction         = make_store(shape);
  const auto redop_kind        = legate::ReductionOpKind::ADD;
  const auto legion_redop_kind = static_cast<std::int32_t>(redop_kind);
  auto input_partition         = task.find_or_declare_partition(input);
  auto output_partition        = task.find_or_declare_partition(output);
  auto reduction_partition     = task.find_or_declare_partition(reduction);
  auto declared_partition      = task.declare_partition();
  auto constraint              = legate::align(input_partition, output_partition);

  ASSERT_EQ(input_partition.impl(), nullptr);
  ASSERT_EQ(output_partition.impl(), nullptr);
  ASSERT_EQ(reduction_partition.impl(), nullptr);
  ASSERT_EQ(declared_partition.impl(), nullptr);
  ASSERT_NE(constraint.impl(), nullptr);
  ASSERT_EQ(task.add_input(input, input_partition).impl(), nullptr);
  ASSERT_EQ(task.add_output(output, output_partition).impl(), nullptr);
  ASSERT_EQ(task.add_reduction(reduction, legion_redop_kind, reduction_partition).impl(), nullptr);
  task.add_constraint(constraint);
  runtime->submit(std::move(task));

  assert_physical_task_result(output, reduction);
}

TEST_F(InlineLaunchUnit, InlinePhysicalTaskMetadata)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto machine  = runtime->get_machine().only(legate::mapping::TaskTarget::CPU);

  if (machine.empty()) {
    GTEST_SKIP() << "Test requires CPU";
  }

  const auto machine_scope    = legate::Scope{machine};
  const auto provenance_scope = legate::Scope{std::string{INLINE_TASK_PROVENANCE}};

  auto task             = create_physical_store_task();
  const auto shape      = legate::Shape{1};
  const auto input      = make_store(shape, PHYSICAL_TASK_INPUT_VALUE);
  const auto output     = runtime->create_store(shape, legate::int64());
  const auto reduction  = make_store(shape);
  const auto redop_kind = legate::ReductionOpKind::ADD;

  ASSERT_EQ(task.provenance(), INLINE_TASK_PROVENANCE);
  task.set_concurrent(true);
  task.set_side_effect(true);
  task.throws_exception(true);
  ASSERT_THAT([&] { task.add_communicator("cpu"); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Communicators are not supported for inline task execution")));

  task.add_input(input);
  task.add_output(output);
  task.add_reduction(reduction, redop_kind);
  runtime->submit(std::move(task));

  assert_physical_task_result(output, reduction);
}

TEST_F(InlineLaunchUnit, Remapping)
{
  constexpr std::size_t REMAP_SIZE   = 8;
  constexpr std::int64_t REMAP_VALUE = 42;

  auto* const runtime = legate::Runtime::get_runtime();
  const auto machine  = runtime->get_machine();

  if (machine.count(legate::mapping::TaskTarget::GPU) == 0) {
    GTEST_SKIP() << "Test requires GPU support";
  }

  const auto lib     = runtime->find_library(Config::LIBRARY_NAME);
  const auto cpu_mem = machine.only(legate::mapping::TaskTarget::CPU);
  const auto gpu_mem = machine.only(legate::mapping::TaskTarget::GPU);

  auto store = runtime->create_store(legate::Shape{REMAP_SIZE}, legate::int64());

  {
    const auto cpu_scope = legate::Scope{cpu_mem};

    runtime->issue_fill(store, legate::Scalar{REMAP_VALUE});
    const auto phys   = store.get_physical_store();
    const auto mdspan = phys.span_read_accessor<std::int64_t, 1>();

    ASSERT_EQ(phys.target(), legate::mapping::StoreTarget::SYSMEM);
    ASSERT_EQ(mdspan.extent(0), REMAP_SIZE);

    for (auto i = 0; i < mdspan.extent(0); ++i) {
      ASSERT_EQ(mdspan(i), REMAP_VALUE)
        << "mdspan(" << i << ") = " << mdspan(i) << " != " << REMAP_VALUE;
    }
  }

  {
    // Perform remapping of store onto GPU memory
    const auto gpu_scope = legate::Scope{gpu_mem};

    auto gpu_task = runtime->create_task(lib, CheckTask::TASK_CONFIG.task_id());
    gpu_task.add_input(store);
    runtime->submit(std::move(gpu_task));
  }

  {
    // Verify that the store values are retained even when remapping back to CPU memory
    const auto cpu_scope = legate::Scope{cpu_mem};

    const auto phys   = store.get_physical_store();
    const auto mdspan = phys.span_read_accessor<std::int64_t, 1>();

    ASSERT_EQ(phys.target(), legate::mapping::StoreTarget::SYSMEM);
    ASSERT_EQ(mdspan.extent(0), REMAP_SIZE);

    for (auto i = 0; i < mdspan.extent(0); ++i) {
      ASSERT_EQ(mdspan(i), REMAP_VALUE)
        << "after remapping: mdspan(" << i << ") = " << mdspan(i) << " != " << REMAP_VALUE;
    }
  }
}

}  // namespace test_inline_launch_basic
