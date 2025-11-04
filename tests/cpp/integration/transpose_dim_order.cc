/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/utilities/detail/type_traits.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>
#include <vector>

namespace transpose_dim_order {

namespace {

constexpr std::string_view LIBRARY_NAME = "transpose_dim_order";
constexpr std::size_t X_LEN             = 10;
constexpr std::size_t Y_LEN             = 20;
constexpr std::size_t Z_LEN             = 30;

template <int NDIM>
legate::LogicalStore make_store()
{
  static_assert(NDIM == 2 || NDIM == 3);

  auto runtime = legate::Runtime::get_runtime();

  if constexpr (NDIM == 2) {
    const auto shape = legate::Shape{X_LEN, Y_LEN};
    return runtime->create_store(shape, legate::int64());
  } else {
    const auto shape = legate::Shape{X_LEN, Y_LEN, Z_LEN};
    return runtime->create_store(shape, legate::int64());
  }
}

template <typename TaskClass>
void launch_task_with_store(const legate::LogicalStore& store, const bool fortran_order)
{
  auto runtime = legate::Runtime::get_runtime();

  auto library = runtime->find_library(LIBRARY_NAME);

  auto task = runtime->create_task(library, TaskClass::TASK_CONFIG.task_id(), {1});
  task.add_output(store);
  task.add_scalar_arg(fortran_order);
  runtime->submit(std::move(task));
}

enum class TestCase : std::uint8_t {
  TRANSPOSE_2D,
  TRANSPOSE_2D_TWICE,
  TRANSPOSE_3D,
  PROMOTE_TRANSPOSE,
  PROJECT_TRANSPOSE,
  DELINEARIZE_TRANSPOSE,
  EMPTY_NO_TRANSFORM,
  EMPTY_PROMOTE_TRANSPOSE,
};

template <typename Derived, TestCase CASE_ID>
class TestTaskBase :  // NOLINT (bugprone-crtp-constructor-accessibility)
                      public legate::LegateTask<TestTaskBase<Derived, CASE_ID>> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{legate::detail::to_underlying(CASE_ID)}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto store               = context.output(0).data();
    const bool fortran_order = context.scalar(0).value<bool>();

    Derived::check(store, fortran_order);
  }
};

class LibraryMapper : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    const auto fortran_order = task.scalar(0).value<bool>();
    auto output              = task.output(0).data();
    auto dim_ordering        = fortran_order ? legate::mapping::DimOrdering::fortran_order()
                                             : legate::mapping::DimOrdering::c_order();

    std::vector<legate::mapping::StoreMapping> mappings{};

    mappings.push_back(legate::mapping::StoreMapping::default_mapping(
      output, options.front(), /*exact*/ true, std::move(dim_ordering)));
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

class TaskTranspose2D : public TestTaskBase<TaskTranspose2D, TestCase::TRANSPOSE_2D> {
 public:
  static void check(const legate::PhysicalStore& store, const bool fortran_order)
  {
    auto store_acc = store.read_accessor<std::int64_t, 2>();
    if (fortran_order) {
      EXPECT_EQ(&(store_acc[{1, 0}]) - &(store_acc[{0, 0}]), 1);
    } else {
      EXPECT_EQ(&(store_acc[{0, 1}]) - &(store_acc[{0, 0}]), 1);
    }
  }
};

class TaskTranspose2DTwice : public TestTaskBase<TaskTranspose2D, TestCase::TRANSPOSE_2D_TWICE> {
 public:
  static void check(const legate::PhysicalStore& store, const bool fortran_order)
  {
    TaskTranspose2D::check(store, fortran_order);
  }
};

class TaskTranspose3D : public TestTaskBase<TaskTranspose3D, TestCase::TRANSPOSE_3D> {
 public:
  static void check(const legate::PhysicalStore& store, const bool fortran_order)
  {
    auto store_acc = store.read_accessor<std::int64_t, 3>();
    if (fortran_order) {
      EXPECT_EQ(&(store_acc[{1, 0, 0}]) - &(store_acc[{0, 0, 0}]), 1);
    } else {
      EXPECT_EQ(&(store_acc[{0, 0, 1}]) - &(store_acc[{0, 0, 0}]), 1);
    }
  }
};

class TaskPromoteTranspose
  : public TestTaskBase<TaskPromoteTranspose, TestCase::PROMOTE_TRANSPOSE> {
 public:
  static void check(const legate::PhysicalStore& store, const bool fortran_order)
  {
    auto store_acc = store.read_accessor<std::int64_t, 3>();
    if (fortran_order) {
      EXPECT_EQ(&(store_acc[{1, 0, 0}]) - &(store_acc[{0, 0, 0}]), 1);
    } else {
      EXPECT_EQ(&(store_acc[{0, 0, 1}]) - &(store_acc[{0, 0, 0}]), 0);
    }
  }
};

class TaskProjectTranspose
  : public TestTaskBase<TaskProjectTranspose, TestCase::PROJECT_TRANSPOSE> {
 public:
  static void check(const legate::PhysicalStore& store, const bool fortran_order)
  {
    auto store_acc = store.read_accessor<std::int64_t, 2>();
    if (fortran_order) {
      EXPECT_EQ(&(store_acc[{0, 1}]) - &(store_acc[{0, 0}]), Z_LEN);
    } else {
      EXPECT_EQ(&(store_acc[{1, 0}]) - &(store_acc[{0, 0}]), X_LEN);
    }
  }
};

class TaskDelinearizeTranspose
  : public TestTaskBase<TaskDelinearizeTranspose, TestCase::DELINEARIZE_TRANSPOSE> {
 public:
  static void check(const legate::PhysicalStore& store, const bool fortran_order)
  {
    auto store_acc = store.read_accessor<std::int64_t, 3>();
    if (fortran_order) {
      EXPECT_EQ(&(store_acc[{1, 0, 0}]) - &(store_acc[{0, 0, 0}]), 1);
    } else {
      EXPECT_EQ(&(store_acc[{0, 0, 1}]) - &(store_acc[{0, 0, 0}]), 1);
    }
  }
};

class TaskEmptyNoTransform
  : public TestTaskBase<TaskEmptyNoTransform, TestCase::EMPTY_NO_TRANSFORM> {
 public:
  static void check(const legate::PhysicalStore& /*store*/, const bool /*fortran_order*/) {}
};

class TaskEmptyPromoteTranspose
  : public TestTaskBase<TaskEmptyPromoteTranspose, TestCase::EMPTY_PROMOTE_TRANSPOSE> {
 public:
  static void check(const legate::PhysicalStore& store, const bool /*fortran_order*/)
  {
    auto store_acc = store.read_accessor<std::int64_t, 2>();
    // distance between elements in both dimensions should be 0 regardless of
    // fortran_order, because we promoted a 0-dim store to a 2-dim store and the
    // dimensions are fake
    EXPECT_EQ(&(store_acc[{1, 0}]) - &(store_acc[{0, 0}]), 0);
  }
};

class TransposeDimOrder : public DefaultFixture, public ::testing::WithParamInterface<bool> {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(
      LIBRARY_NAME, legate::ResourceConfig{}, std::make_unique<LibraryMapper>(), {}, &created);
    if (created) {
      TaskTranspose2D::register_variants(library);
      TaskTranspose2DTwice::register_variants(library);
      TaskTranspose3D::register_variants(library);
      TaskPromoteTranspose::register_variants(library);
      TaskProjectTranspose::register_variants(library);
      TaskDelinearizeTranspose::register_variants(library);
      TaskEmptyNoTransform::register_variants(library);
      TaskEmptyPromoteTranspose::register_variants(library);
    }
  }
};

INSTANTIATE_TEST_SUITE_P(TransposeDimOrder, TransposeDimOrder, ::testing::Values(false, true));

TEST_P(TransposeDimOrder, Transpose2D)
{
  auto store               = make_store<2>();
  const bool fortran_order = GetParam();
  store                    = store.transpose({1, 0});
  launch_task_with_store<TaskTranspose2D>(store, fortran_order);
}

TEST_P(TransposeDimOrder, Transpose2DTwice)
{
  auto store               = make_store<2>();
  const bool fortran_order = GetParam();
  store                    = store.transpose({1, 0}).transpose({1, 0});
  launch_task_with_store<TaskTranspose2DTwice>(store, fortran_order);
}

TEST_P(TransposeDimOrder, Transpose3D)
{
  auto store               = make_store<3>();
  const bool fortran_order = GetParam();
  store                    = store.transpose({2, 1, 0});
  launch_task_with_store<TaskTranspose3D>(store, fortran_order);
}

TEST_P(TransposeDimOrder, PromoteTranspose)
{
  auto store               = make_store<2>();
  const bool fortran_order = GetParam();
  store                    = store.promote(1, Z_LEN).transpose({0, 2, 1});
  launch_task_with_store<TaskPromoteTranspose>(store, fortran_order);
}

TEST_P(TransposeDimOrder, ProjectTranspose)
{
  auto store               = make_store<3>();
  const bool fortran_order = GetParam();
  store                    = store.project(1, 0).transpose({1, 0});
  launch_task_with_store<TaskProjectTranspose>(store, fortran_order);
}

TEST_P(TransposeDimOrder, DelinearizeTranspose)
{
  auto store               = make_store<2>();
  const bool fortran_order = GetParam();
  // factorize Y_LEN into two factors
  constexpr std::size_t FACTOR_A = 4;
  constexpr std::size_t FACTOR_B = Y_LEN / FACTOR_A;
  ASSERT_EQ(Y_LEN, FACTOR_A * FACTOR_B);
  store = store.delinearize(1, {FACTOR_A, FACTOR_B}).transpose({2, 1, 0});
  launch_task_with_store<TaskDelinearizeTranspose>(store, fortran_order);
}

TEST_P(TransposeDimOrder, EmptyNoTransform)
{
  auto runtime             = legate::Runtime::get_runtime();
  auto store               = runtime->create_store(legate::Shape{}, legate::int64());
  const bool fortran_order = GetParam();
  launch_task_with_store<TaskEmptyNoTransform>(store, fortran_order);
}

TEST_P(TransposeDimOrder, EmptyPromoteTranspose)
{
  auto runtime             = legate::Runtime::get_runtime();
  auto store               = runtime->create_store(legate::Shape{}, legate::int64());
  const bool fortran_order = GetParam();
  store                    = store.promote(0, X_LEN).promote(1, Y_LEN).transpose({1, 0});
  launch_task_with_store<TaskEmptyPromoteTranspose>(store, fortran_order);
}

}  // namespace transpose_dim_order
