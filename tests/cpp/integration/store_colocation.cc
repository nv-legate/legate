/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace store_colocation_test {

namespace {

class ColocationTestTask : public legate::LegateTask<ColocationTestTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    for (auto& store : outputs) {
      if (store.is_unbound_store()) {
        store.bind_empty_data();
      }
    }
  }
};

class ReductionTestTask : public legate::LegateTask<ReductionTestTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto outputs    = context.outputs();
    auto reductions = context.reductions();

    for (auto& store : outputs) {
      if (store.is_unbound_store()) {
        store.bind_empty_data();
      }
    }

    for (auto& store : reductions) {
      if (store.is_unbound_store()) {
        store.bind_empty_data();
      }
    }
  }
};

// can_colocate_with() behavior for unbound vs unbound stores
class UnboundColocationMapper : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task&, const std::vector<legate::mapping::StoreTarget>&) override
  {
    return {};
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task& /*task*/,
                                                  legate::mapping::StoreTarget /*memory*/) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    return legate::Scalar{};
  }
};

// can_colocate_with() behavior for bound vs unbound stores and bound vs bound stores
class BoundColocationMapper : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task&, const std::vector<legate::mapping::StoreTarget>&) override
  {
    return {};
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task& /*task*/,
                                                  legate::mapping::StoreTarget /*memory*/) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    return legate::Scalar{};
  }
};

// can_colocate_with() behavior for reduction stores and normal vs reduction stores
class ReductionColocationMapper : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task&, const std::vector<legate::mapping::StoreTarget>&) override
  {
    return {};
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task& /*task*/,
                                                  legate::mapping::StoreTarget /*memory*/) override
  {
    return std::nullopt;
  }

  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    return legate::Scalar{};
  }
};

// Config for unbound colocation tests
class UnboundColocationConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = "store_colocation_unbound_test";

  static void registration_callback(legate::Library library)
  {
    ColocationTestTask::register_variants(library);
  }

  static std::unique_ptr<legate::mapping::Mapper> create_mapper()
  {
    return std::make_unique<UnboundColocationMapper>();
  }
};

// Config for bound colocation tests
class BoundColocationConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = "store_colocation_bound_test";

  static void registration_callback(legate::Library library)
  {
    ColocationTestTask::register_variants(library);
  }

  static std::unique_ptr<legate::mapping::Mapper> create_mapper()
  {
    return std::make_unique<BoundColocationMapper>();
  }
};

// Config for reduction colocation tests
class ReductionColocationConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = "store_colocation_reduction_test";

  static void registration_callback(legate::Library library)
  {
    ColocationTestTask::register_variants(library);
    ReductionTestTask::register_variants(library);
  }

  static std::unique_ptr<legate::mapping::Mapper> create_mapper()
  {
    return std::make_unique<ReductionColocationMapper>();
  }
};

// Base test fixtures for each library with custom mappers
class UnboundColocationFixture : public DefaultFixture {
 protected:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(UnboundColocationConfig::LIBRARY_NAME,
                                                   legate::ResourceConfig{},
                                                   UnboundColocationConfig::create_mapper(),
                                                   {},
                                                   &created);
    if (created) {
      UnboundColocationConfig::registration_callback(library);
    }
  }

  legate::Library get_library_()
  {
    return legate::Runtime::get_runtime()->find_library(UnboundColocationConfig::LIBRARY_NAME);
  }
};

class BoundColocationFixture : public DefaultFixture {
 protected:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(BoundColocationConfig::LIBRARY_NAME,
                                                   legate::ResourceConfig{},
                                                   BoundColocationConfig::create_mapper(),
                                                   {},
                                                   &created);
    if (created) {
      BoundColocationConfig::registration_callback(library);
    }
  }

  legate::Library get_library_()
  {
    return legate::Runtime::get_runtime()->find_library(BoundColocationConfig::LIBRARY_NAME);
  }
};

class ReductionColocationFixture : public DefaultFixture {
 protected:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(ReductionColocationConfig::LIBRARY_NAME,
                                                   legate::ResourceConfig{},
                                                   ReductionColocationConfig::create_mapper(),
                                                   {},
                                                   &created);
    if (created) {
      ReductionColocationConfig::registration_callback(library);
    }
  }

  legate::Library get_library_()
  {
    return legate::Runtime::get_runtime()->find_library(ReductionColocationConfig::LIBRARY_NAME);
  }
};

}  // namespace

TEST_F(UnboundColocationFixture, UnboundWithUnbound)
{
  auto* runtime = legate::Runtime::get_runtime();
  auto library  = get_library_();
  auto task     = runtime->create_task(library, ColocationTestTask::TASK_CONFIG.task_id());

  // Create two unbound stores
  auto unbound1 = runtime->create_store(legate::int32(), /*dim=*/1);
  auto unbound2 = runtime->create_store(legate::float32(), /*dim=*/1);

  ASSERT_TRUE(unbound1.unbound());
  ASSERT_TRUE(unbound2.unbound());

  task.add_output(unbound1);
  task.add_output(unbound2);
  runtime->submit(std::move(task));
  runtime->issue_execution_fence();
}

TEST_F(BoundColocationFixture, BoundWithUnbound)
{
  auto* runtime      = legate::Runtime::get_runtime();
  auto library       = get_library_();
  auto task          = runtime->create_task(library, ColocationTestTask::TASK_CONFIG.task_id());
  auto bound_store   = runtime->create_store(legate::Shape{8}, legate::int32());
  auto unbound_store = runtime->create_store(legate::float32(), /*dim=*/1);

  ASSERT_FALSE(bound_store.unbound());
  ASSERT_TRUE(unbound_store.unbound());

  task.add_output(bound_store);
  task.add_output(unbound_store);
  runtime->submit(std::move(task));
  runtime->issue_execution_fence();
}

TEST_F(ReductionColocationFixture, ReductionStores)
{
  auto* runtime = legate::Runtime::get_runtime();
  auto library  = get_library_();

  // Create reduction stores
  auto red1 = runtime->create_store(legate::Shape{3}, legate::int32());
  auto red2 = runtime->create_store(legate::Shape{8}, legate::int32());

  // Initialize first
  auto init_task = runtime->create_task(library, ColocationTestTask::TASK_CONFIG.task_id());
  init_task.add_output(red1);
  init_task.add_output(red2);
  runtime->submit(std::move(init_task));

  // Test reduction colocation
  auto task = runtime->create_task(library, ReductionTestTask::TASK_CONFIG.task_id());
  task.add_reduction(red1, legate::ReductionOpKind::ADD);
  task.add_reduction(red2, legate::ReductionOpKind::ADD);
  runtime->submit(std::move(task));
  runtime->issue_execution_fence();
}

TEST_F(ReductionColocationFixture, NormalWithReduction)
{
  auto* runtime        = legate::Runtime::get_runtime();
  auto library         = get_library_();
  auto normal_store    = runtime->create_store(legate::Shape{7}, legate::int32());
  auto reduction_store = runtime->create_store(legate::Shape{2}, legate::int32());

  // Initialize reduction store
  auto init_task = runtime->create_task(library, ColocationTestTask::TASK_CONFIG.task_id());
  init_task.add_output(reduction_store);
  runtime->submit(std::move(init_task));

  // Test normal + reduction colocation
  auto task = runtime->create_task(library, ReductionTestTask::TASK_CONFIG.task_id());
  task.add_output(normal_store);
  task.add_reduction(reduction_store, legate::ReductionOpKind::ADD);
  runtime->submit(std::move(task));
  runtime->issue_execution_fence();
}

}  // namespace store_colocation_test
