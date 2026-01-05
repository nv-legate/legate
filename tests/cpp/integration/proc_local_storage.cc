/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/proc_local_storage.h>

#include <legate.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace proc_local_storage {

namespace {

constexpr std::uint64_t EXTENT = 42;

class Handle {
 public:
  explicit Handle(const legate::DomainPoint& point) : point_{point} {}

  [[nodiscard]] const legate::DomainPoint& point() const { return point_; }

 private:
  legate::DomainPoint point_{};
};

class PrimitiveTester : public legate::LegateTask<PrimitiveTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    static legate::ProcLocalStorage<std::int64_t> storage{};

    if (!storage.has_value()) {
      storage.emplace(context.get_task_index()[0]);
    } else {
      EXPECT_EQ(storage.get(), context.get_task_index()[0]);
    }
  }
};

class ObjectTester : public legate::LegateTask<ObjectTester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext context)
  {
    static legate::ProcLocalStorage<Handle> storage{};

    if (!storage.has_value()) {
      storage.emplace(context.get_task_index());
    } else {
      EXPECT_EQ(storage.get().point(), context.get_task_index());
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_proc_local_storage";

  static void registration_callback(legate::Library library)
  {
    PrimitiveTester::register_variants(library);
    ObjectTester::register_variants(library);
  }
};

class ProcLocalStorage : public RegisterOnceFixture<Config> {};

void run_test(legate::LocalTaskID task_id)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  // Dummy store to get the task parallelized
  auto store = runtime->create_store(legate::Shape{EXTENT}, legate::int64());

  auto setter = runtime->create_task(library, task_id);
  setter.add_output(store);
  runtime->submit(std::move(setter));

  auto getter = runtime->create_task(library, task_id);
  getter.add_input(store);
  runtime->submit(std::move(getter));
}

void test_uninitialized()
{
  legate::ProcLocalStorage<std::int64_t> storage{};
  EXPECT_THROW(static_cast<void>(storage.get()), std::logic_error);
}

}  // namespace

TEST_F(ProcLocalStorage, Primitive) { run_test(PrimitiveTester::TASK_CONFIG.task_id()); }

TEST_F(ProcLocalStorage, Object) { run_test(ObjectTester::TASK_CONFIG.task_id()); }

TEST_F(ProcLocalStorage, UninitializedAccess) { test_uninitialized(); }

}  // namespace proc_local_storage
