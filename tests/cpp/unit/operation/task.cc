/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/task.h>

#include <legate.h>

#include <legate/operation/detail/task_launcher.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/task_info.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utilities/utilities.h>
#include <variant>

namespace operation_task_test {

namespace {

class ToStringTask : public legate::LegateTask<ToStringTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class CommunicatorTask : public legate::LegateTask<CommunicatorTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_variant_options(
      legate::VariantOptions{}.with_communicators({"cpu"}));

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_operation_task";

  static void registration_callback(legate::Library library)
  {
    ToStringTask::register_variants(library);
    CommunicatorTask::register_variants(library);
  }
};

class TaskBaseUnit : public RegisterOnceFixture<Config> {};

class TaskBaseDeathTest : public TaskBaseUnit {};

// This concrete TaskBase keeps the default add_constraint() and add_communicator()
// implementations so tests can exercise the unsupported fallback paths.
class UnsupportedTaskBase final : public legate::detail::TaskBase {
 public:
  UnsupportedTaskBase(const legate::detail::Library& library,
                      const legate::detail::VariantInfo& variant_info)
    : TaskBase{library,
               variant_info,
               ToStringTask::TASK_CONFIG.task_id(),
               BASE_TASK_UNIQUE_ID,
               BASE_TASK_PRIORITY,
               legate::mapping::detail::Machine{}}
  {
  }

  [[nodiscard]] const legate::detail::Variable* add_input(
    legate::InternalSharedPtr<legate::detail::LogicalStore> /*store*/) override
  {
    return nullptr;
  }

  [[nodiscard]] const legate::detail::Variable* add_output(
    legate::InternalSharedPtr<legate::detail::LogicalStore> /*store*/) override
  {
    return nullptr;
  }

  [[nodiscard]] const legate::detail::Variable* add_reduction(
    legate::InternalSharedPtr<legate::detail::LogicalStore> /*store*/,
    std::int32_t /*redop_kind*/) override
  {
    return nullptr;
  }

  void add_input(legate::InternalSharedPtr<legate::detail::LogicalStore> /*store*/,
                 const legate::detail::Variable* /*partition_symbol*/) override
  {
  }

  void add_output(legate::InternalSharedPtr<legate::detail::LogicalStore> /*store*/,
                  const legate::detail::Variable* /*partition_symbol*/) override
  {
  }

  void add_reduction(legate::InternalSharedPtr<legate::detail::LogicalStore> /*store*/,
                     std::int32_t /*redop_kind*/,
                     const legate::detail::Variable* /*partition_symbol*/) override
  {
  }

  [[nodiscard]] Kind kind() const override { return Kind::AUTO_TASK; }

  [[nodiscard]] bool needs_partitioning() const override { return false; }

 private:
  static constexpr auto BASE_TASK_UNIQUE_ID = std::uint64_t{0};
  static constexpr auto BASE_TASK_PRIORITY  = std::int32_t{0};
};

[[nodiscard]] UnsupportedTaskBase create_unsupported_task_base()
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task_info      = library.find_task(ToStringTask::TASK_CONFIG.task_id());
  const auto variant  = task_info.impl()->find_variant(legate::VariantCode::CPU);

  if (!variant) {
    throw std::runtime_error{"CPU variant is not registered"};
  }

  return UnsupportedTaskBase{*library.impl(), variant->get()};
}

constexpr auto MANUAL_TASK_UNIQUE_ID   = std::uint64_t{1};
constexpr auto MANUAL_TASK_PRIORITY    = std::int32_t{0};
constexpr auto PHYSICAL_TASK_UNIQUE_ID = std::uint64_t{2};

[[nodiscard]] legate::detail::ManualTask create_manual_task(const legate::Domain& domain)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task_info      = library.find_task(ToStringTask::TASK_CONFIG.task_id());
  const auto variant  = task_info.impl()->find_variant(legate::VariantCode::CPU);

  if (!variant) {
    throw std::runtime_error{"CPU variant is not registered"};
  }

  return legate::detail::ManualTask{*library.impl(),
                                    variant->get(),
                                    ToStringTask::TASK_CONFIG.task_id(),
                                    domain,
                                    MANUAL_TASK_UNIQUE_ID,
                                    MANUAL_TASK_PRIORITY,
                                    legate::mapping::detail::Machine{}};
}

[[nodiscard]] legate::detail::ManualTask create_manual_task()
{
  const auto domain = legate::Domain{legate::Rect<1>{0, 1}};

  return create_manual_task(domain);
}

[[nodiscard]] legate::detail::PhysicalTask create_physical_task()
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task_info      = library.find_task(ToStringTask::TASK_CONFIG.task_id());
  const auto variant  = task_info.impl()->find_variant(legate::VariantCode::CPU);

  if (!variant) {
    throw std::runtime_error{"CPU variant is not registered"};
  }

  return legate::detail::PhysicalTask{*library.impl(),
                                      variant->get(),
                                      ToStringTask::TASK_CONFIG.task_id(),
                                      PHYSICAL_TASK_UNIQUE_ID,
                                      legate::mapping::detail::Machine{}};
}

}  // namespace

TEST_F(TaskBaseUnit, ToStringIncludesProvenance)
{
  const auto provenance = std::string{"task-base-to-string-provenance"};
  auto* const runtime   = legate::Runtime::get_runtime();
  auto library          = runtime->find_library(Config::LIBRARY_NAME);
  const auto scope      = legate::Scope{provenance};
  auto task             = runtime->create_task(library, ToStringTask::TASK_CONFIG.task_id());

  const auto base = task.impl_()->to_string(/*show_provenance=*/false);
  auto expected   = base;

  expected.append("[").append(provenance).append("]");
  ASSERT_EQ(task.impl_()->to_string(/*show_provenance=*/true), expected);
}

TEST_F(TaskBaseUnit, CreateTaskWithPredeclaredCommunicator)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, CommunicatorTask::TASK_CONFIG.task_id());

  ASSERT_EQ(task.impl_()->kind(), legate::detail::Operation::Kind::AUTO_TASK);
}

TEST_F(TaskBaseUnit, SubmittedAutoTaskDoesNotNeedInlineExecution)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, ToStringTask::TASK_CONFIG.task_id());

  ASSERT_FALSE(task.needs_inline_execution_());

  runtime->submit(std::move(task));

  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_FALSE(task.needs_inline_execution_());
}

TEST_F(TaskBaseUnit, SubmittedManualTaskCannotBeReused)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, ToStringTask::TASK_CONFIG.task_id(), {1});

  runtime->submit(std::move(task));

  // Accessors route through ManualTask::impl_(), which should reject descriptors that have already
  // been submitted.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_THAT([&] { static_cast<void>(task.provenance()); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Illegal to reuse task descriptors")));
}

TEST_F(TaskBaseUnit, SubmittedManualTaskCannotBeSubmittedAgain)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, ToStringTask::TASK_CONFIG.task_id(), {1});

  runtime->submit(std::move(task));

  // A repeated submit routes through ManualTask::release_(), so it must fail before returning a
  // null operation to the detail runtime.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_THAT([&] { runtime->submit(std::move(task)); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Illegal to reuse task descriptors")));
}

TEST_F(TaskBaseUnit, AddCommunicatorThrowsForPredeclaredCommunicator)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);
  auto task           = runtime->create_task(library, CommunicatorTask::TASK_CONFIG.task_id());

  ASSERT_THAT([&] { task.add_communicator("cpu"); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("has pre-declared communicator")));
}

TEST_F(TaskBaseUnit, ConcurrentTaskThrowsInsideStreamingScope)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->find_library(Config::LIBRARY_NAME);

  ASSERT_THAT(
    [&] {
      auto scope =
        legate::Scope{legate::ParallelPolicy{}.with_streaming(legate::StreamingMode::RELAXED)};
      auto task = runtime->create_task(library, ToStringTask::TASK_CONFIG.task_id());

      task.set_concurrent(true);
      runtime->submit(std::move(task));
    },
    ::testing::ThrowsMessage<std::runtime_error>(
      ::testing::HasSubstr("Concurrent Tasks are not allowed inside a Streaming Scope")));
}

TEST_F(TaskBaseDeathTest, InterferingStoresAbortWithTaskName)
{
  auto* const runtime    = legate::Runtime::get_runtime();
  auto library           = runtime->find_library(Config::LIBRARY_NAME);
  auto store             = runtime->create_store(legate::Shape{1}, legate::int32());
  const auto& store_impl = store.impl();
  auto* const store_ptr  = store_impl.get();
  const auto input_proj  = legate::detail::StoreProjection{};
  auto output_proj       = legate::detail::StoreProjection{};
  auto launcher          = legate::detail::TaskLauncher{*library.impl(),
                                                        runtime->impl()->get_machine(),
                                                        legate::ParallelPolicy{},
                                                        ToStringTask::TASK_CONFIG.task_id(),
                                                        Legion::ProjectionID{0}};

  output_proj.proj_id = Legion::ProjectionID{1};

  launcher.add_input(legate::detail::RegionFieldArg{store_ptr, LEGION_READ_ONLY, input_proj});
  launcher.add_output(legate::detail::RegionFieldArg{store_ptr, LEGION_WRITE_ONLY, output_proj});

  ASSERT_DEATH(static_cast<void>(launcher.execute_single()),
               "Task .* has interfering store arguments");
}

TEST_F(TaskBaseUnit, AddConstraintThrowsUnsupported)
{
  auto task = create_unsupported_task_base();

  // UnsupportedTaskBase does not override add_constraint(), so this dispatches to TaskBase's
  // unsupported fallback.
  ASSERT_THAT([&] { task.add_constraint({}, /*bypass_signature_check=*/false); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("add_constraint not supported for this task type")));
}

TEST_F(TaskBaseUnit, AddCommunicatorThrowsUnsupported)
{
  auto task = create_unsupported_task_base();

  // UnsupportedTaskBase does not override add_communicator(), so this dispatches to TaskBase's
  // unsupported fallback.
  ASSERT_THAT([&] { task.add_communicator("test-communicator", /*bypass_signature_check=*/false); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("add_communicator not supported for this task type")));
}

TEST_F(TaskBaseUnit, PhysicalTaskKindAndPartitioning)
{
  auto task = create_physical_task();

  ASSERT_EQ(task.kind(), legate::detail::Operation::Kind::PHYSICAL_TASK);
  ASSERT_FALSE(task.needs_partitioning());
}

TEST_F(TaskBaseUnit, ManualTaskAddScalarPartitionReductionRecordsScalarReduction)
{
  auto task           = create_manual_task();
  auto* const runtime = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Scalar{std::int32_t{1}, legate::int32()});
  auto partition      = store.partition_by_tiling({1});
  constexpr auto redop_kind = legate::ReductionOpKind::ADD;
  const auto store_impl     = legate::InternalSharedPtr<legate::detail::LogicalStore>{store.impl()};
  const auto legion_redop_id = store.impl()->type()->find_reduction_operator(redop_kind);

  task.add_reduction(partition.impl(),
                     static_cast<std::int32_t>(redop_kind),
                     std::nullopt,
                     /*is_key_partition=*/false);

  ASSERT_EQ(task.reductions().size(), 1);
  ASSERT_EQ(task.scalar_reductions().size(), 1);
  ASSERT_EQ(task.scalar_reductions().at(0).first, store_impl);
  ASSERT_EQ(task.scalar_reductions().at(0).second, legion_redop_id);
}

TEST_F(TaskBaseUnit, ManualTaskAddStoreWithInvalidLaunchDomainSkipsProjection)
{
  auto task             = create_manual_task(legate::Domain{});
  auto* const runtime   = legate::Runtime::get_runtime();
  auto store            = runtime->create_store(legate::Shape{1}, legate::int32());
  const auto store_impl = legate::InternalSharedPtr<legate::detail::LogicalStore>{store.impl()};

  const auto* partition_symbol = task.add_input(store_impl);

  ASSERT_EQ(partition_symbol, nullptr);
  ASSERT_FALSE(task.launch_domain().is_valid());
  ASSERT_EQ(task.inputs().size(), 1);
  ASSERT_NE(task.inputs().at(0).variable, nullptr);
  ASSERT_EQ(
    std::get<legate::InternalSharedPtr<legate::detail::LogicalStore>>(task.inputs().at(0).store),
    store_impl);
}

TEST_F(TaskBaseUnit, ManualTaskAddInputWithPartitionSymbolThrowsUnsupported)
{
  auto task = create_manual_task();

  ASSERT_THAT([&] { task.add_input({}, nullptr); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Not supported for ManualTask")));
}

TEST_F(TaskBaseUnit, ManualTaskAddOutputWithPartitionSymbolThrowsUnsupported)
{
  auto task = create_manual_task();

  ASSERT_THAT([&] { task.add_output({}, nullptr); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Not supported for ManualTask")));
}

TEST_F(TaskBaseUnit, ManualTaskAddReductionWithPartitionSymbolThrowsUnsupported)
{
  auto task = create_manual_task();

  ASSERT_THAT([&] { task.add_reduction({}, /*redop_kind=*/0, nullptr); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Not supported for ManualTask")));
}

}  // namespace operation_task_test
