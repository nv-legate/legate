/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace test_projection_c_order {

class Init : public legate::LegateTask<Init> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto acc1 = context.output(0).data().write_accessor<std::int64_t, 3>();
    auto acc2 =
      context.output(1).data().write_accessor<std::int64_t*, 1, /* VALIDATE_TYPE */ false>();

    acc2[0] = acc1.ptr(legate::Point<3>{0, 0, 0});
  }
};

class Tester : public legate::LegateTask<Tester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto acc1 = context.input(0).data().read_accessor<std::int64_t, 2>();
    auto acc2 =
      context.input(1).data().read_accessor<std::int64_t*, 1, /* VALIDATE_TYPE */ false>();

    ASSERT_EQ(acc2[0], acc1.ptr(legate::Point<2>{0, 0}));
  }
};

class Tester2 : public legate::LegateTask<Tester2> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto store     = context.output(0).data();
    auto store_acc = store.read_accessor<std::int64_t, 2>();
    EXPECT_EQ(&(store_acc[{0, 1}]) - &(store_acc[{0, 0}]), 1);
  }
};

namespace {

constexpr std::string_view LIBRARY_NAME = "test_projection_c_order";

class LibraryMapper : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    if (task.task_id() != Tester2::TASK_CONFIG.task_id()) {
      return {};
    }

    std::vector<legate::mapping::StoreMapping> mappings;

    mappings.push_back(
      legate::mapping::StoreMapping::default_mapping(task.output(0).data(),
                                                     options.front(),
                                                     /*exact*/ true,
                                                     legate::mapping::DimOrdering::c_order()));

    return mappings;
  }

  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    return legate::Scalar{};
  }

  std::optional<std::size_t> allocation_pool_size(const legate::mapping::Task&,
                                                  legate::mapping::StoreTarget) override
  {
    return std::nullopt;
  }
};

}  // namespace

class ProjectionCOrder : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();

    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(
      LIBRARY_NAME, legate::ResourceConfig{}, std::make_unique<LibraryMapper>(), {}, &created);

    if (created) {
      Init::register_variants(library);
      Tester::register_variants(library);
      Tester2::register_variants(library);
    }
  }
};

TEST_F(ProjectionCOrder, AliasSharingInstance)
{
  constexpr int X = 100;
  constexpr int Y = 1;
  constexpr int Z = 100;
  auto runtime    = legate::Runtime::get_runtime();
  auto shape      = legate::Shape{X, Y, Z};
  auto store1     = runtime->create_store(shape, legate::int64());
  auto store2     = runtime->create_store(
    {1}, legate::binary_type(sizeof(std::int64_t*)), /*optimize_scalar=*/true);

  auto library = runtime->find_library(LIBRARY_NAME);
  {
    auto task = runtime->create_task(library, Init::TASK_CONFIG.task_id(), {1});
    task.add_output(store1);
    task.add_output(store2);
    runtime->submit(std::move(task));
  }
  {
    auto task = runtime->create_task(library, Tester::TASK_CONFIG.task_id(), {1});
    task.add_input(store1.project(/*dim=*/1, /*index=*/0));
    task.add_input(store2);
    runtime->submit(std::move(task));
  }
}

TEST_F(ProjectionCOrder, TransposeFollowedByProjection)
{
  constexpr int X = 3;
  constexpr int Y = 4;
  constexpr int Z = 5;
  auto runtime    = legate::Runtime::get_runtime();
  auto shape      = legate::Shape{X, Y, Z};
  auto store      = runtime->create_store(shape, legate::int64());

  auto library = runtime->find_library(LIBRARY_NAME);
  auto task    = runtime->create_task(library, Tester2::TASK_CONFIG.task_id(), {1});
  task.add_output(store.transpose({2, 0, 1}).project(/*dim=*/0, /*index=*/1));
  runtime->submit(std::move(task));
}

}  // namespace test_projection_c_order
