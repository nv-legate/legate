/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_transpose_c_order {

class Tester : public legate::LegateTask<Tester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto store     = context.output(0).data();
    auto store_acc = store.read_accessor<std::int64_t, 3>();
    EXPECT_EQ(&(store_acc[{0, 0, 1}]) - &(store_acc[{0, 0, 0}]), 1);
  }
};

namespace {

constexpr std::string_view LIBRARY_NAME = "test_transpose_c_order";

class LibraryMapper : public legate::mapping::Mapper {
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
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

class TransposeCOrder : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();

    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(
      LIBRARY_NAME, legate::ResourceConfig{}, std::make_unique<LibraryMapper>(), {}, &created);

    if (created) {
      Tester::register_variants(library);
    }
  }
};

TEST_F(TransposeCOrder, Test)
{
  constexpr int X = 3;
  constexpr int Y = 4;
  constexpr int Z = 5;
  auto runtime    = legate::Runtime::get_runtime();
  auto shape      = legate::Shape{X, Y, Z};
  auto store      = runtime->create_store(shape, legate::int64());

  auto library = runtime->find_library(LIBRARY_NAME);
  auto task    = runtime->create_task(library, Tester::TASK_CONFIG.task_id(), {1});
  task.add_output(store.transpose({2, 0, 1}));
  runtime->submit(std::move(task));
}

}  // namespace test_transpose_c_order
