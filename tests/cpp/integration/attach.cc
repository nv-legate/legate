/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace attach {

namespace {

[[nodiscard]] const legate::tuple<std::uint64_t>& SHAPE_1D()
{
  static const legate::tuple<std::uint64_t> shape{5};

  return shape;
}

[[nodiscard]] const legate::tuple<std::uint64_t>& SHAPE_2D()
{
  static const legate::tuple<std::uint64_t> shape{3, 4};

  return shape;
}

void increment_physical_store(const legate::PhysicalStore& store, std::int32_t dim)
{
  if (dim == 1) {
    auto shape = store.shape<1>();
    auto acc   = store.read_write_accessor<std::int64_t, 1, true>(shape);
    for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
      acc[*it] += 1;
    }
  } else {
    auto shape = store.shape<2>();
    auto acc   = store.read_write_accessor<std::int64_t, 2, true>(shape);
    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      acc[*it] += 1;
    }
  }
}

void check_physical_store(const legate::PhysicalStore& store,
                          std::int32_t dim,
                          std::int64_t counter)
{
  if (dim == 1) {
    auto shape = store.shape<1>();
    auto acc   = store.read_accessor<std::int64_t, 1, true>(shape);
    for (std::size_t i = 0; i < SHAPE_1D()[0]; ++i) {
      EXPECT_EQ(acc[i], counter++);
    }
  } else {
    auto shape = store.shape<2>();
    auto acc   = store.read_accessor<std::int64_t, 2, true>(shape);
    // Legate should always see elements in the expected order
    for (std::uint64_t i = 0; i < SHAPE_2D()[0]; ++i) {
      for (std::uint64_t j = 0; j < SHAPE_2D()[1]; ++j) {
        const auto p =
          legate::Point<2>{static_cast<legate::coord_t>(i), static_cast<legate::coord_t>(j)};

        EXPECT_EQ(acc[p], counter);
        ++counter;
      }
    }
  }
}

enum TaskOpCode : std::uint8_t { ADDER = 0, CHECKER = 1 };

struct AdderTask : public legate::LegateTask<AdderTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{ADDER}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto output    = context.output(0).data();
    const auto dim = context.scalar(0).value<std::int32_t>();
    increment_physical_store(output, dim);
  }
};

struct CheckerTask : public legate::LegateTask<CheckerTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CHECKER}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto input         = context.input(0).data();
    const auto dim     = context.scalar(0).value<std::int32_t>();
    const auto counter = context.scalar(1).value<std::int64_t>();
    check_physical_store(input, dim, counter);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_attach";

  static void registration_callback(legate::Library library)
  {
    AdderTask::register_variants(library);
    CheckerTask::register_variants(library);
  }
};

class Attach : public RegisterOnceFixture<Config> {};

class Positive : public RegisterOnceFixture<Config>,
                 public ::testing::WithParamInterface<
                   std::tuple<std::pair<std::int32_t, bool>, bool, bool, bool, bool>> {};

INSTANTIATE_TEST_SUITE_P(Attach,
                         Positive,
                         ::testing::Combine(::testing::Values(std::make_pair(1, false),
                                                              std::make_pair(2, false),
                                                              std::make_pair(2, true)),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

std::int64_t* make_buffer(std::int32_t dim, bool fortran)
{
  std::int64_t* buffer;
  std::int64_t counter = 0;
  if (dim == 1) {
    buffer = new std::int64_t[SHAPE_1D().volume()];
    for (std::size_t i = 0; i < SHAPE_1D()[0]; ++i) {
      buffer[i] = counter++;
    }
  } else {
    buffer = new std::int64_t[SHAPE_2D().volume()];
    for (std::size_t i = 0; i < SHAPE_2D()[0]; ++i) {
      for (std::size_t j = 0; j < SHAPE_2D()[1]; ++j) {
        if (fortran) {
          buffer[(j * SHAPE_2D()[0]) + i] = counter++;
        } else {
          buffer[(i * SHAPE_2D()[1]) + j] = counter++;
        }
      }
    }
  }
  return buffer;
}

void check_buffer(std::int64_t* buffer, std::int32_t dim, bool fortran, std::int64_t counter)
{
  if (dim == 1) {
    for (std::size_t i = 0; i < SHAPE_1D()[0]; ++i) {
      EXPECT_EQ(buffer[i], counter++);
    }
  } else {
    for (std::size_t i = 0; i < SHAPE_2D()[0]; ++i) {
      for (std::size_t j = 0; j < SHAPE_2D()[1]; ++j) {
        if (fortran) {
          EXPECT_EQ(buffer[(j * SHAPE_2D()[0]) + i], counter);
        } else {
          EXPECT_EQ(buffer[(i * SHAPE_2D()[1]) + j], counter);
        }
        ++counter;
      }
    }
  }
}

void test_body(
  std::int32_t dim, bool fortran, bool unordered, bool read_only, bool use_tasks, bool use_inline)
{
  auto runtime         = legate::Runtime::get_runtime();
  auto context         = runtime->find_library(Config::LIBRARY_NAME);
  std::int64_t counter = 0;
  auto buffer          = make_buffer(dim, fortran);
  auto l_store         = runtime->create_store(dim == 1 ? SHAPE_1D() : SHAPE_2D(),
                                       legate::int64(),
                                       buffer,
                                       read_only,
                                       fortran ? legate::mapping::DimOrdering::fortran_order()
                                               : legate::mapping::DimOrdering::c_order());
  if (unordered) {
    l_store.impl()->allow_out_of_order_destruction();
  }
  if (read_only) {
    check_buffer(buffer, dim, fortran, counter);
  }
  for (auto iter = 0; iter < 2; ++iter) {
    if (use_tasks) {
      auto task = runtime->create_task(context, AdderTask::TASK_CONFIG.task_id(), {1});
      task.add_input(l_store);
      task.add_output(l_store);
      task.add_scalar_arg(legate::Scalar{dim});
      runtime->submit(std::move(task));
      ++counter;
    }
    if (use_inline) {
      auto p_store = l_store.get_physical_store();
      increment_physical_store(p_store, dim);
      ++counter;
    }
  }
  if (use_tasks) {
    auto task = runtime->create_task(context, CheckerTask::TASK_CONFIG.task_id(), {1});
    task.add_input(l_store);
    task.add_scalar_arg(legate::Scalar{dim});
    task.add_scalar_arg(legate::Scalar{counter});
    runtime->submit(std::move(task));
  }
  if (use_inline) {
    auto p_store = l_store.get_physical_store();
    check_physical_store(p_store, dim, counter);
  }
  l_store.detach();
  if (!read_only) {
    check_buffer(buffer, dim, fortran, counter);
  }
  // Legate no longer copies read-only attachments, so the only safe point to deallocate them is
  // after they are detached from the stores
  delete[] buffer;
}

}  // namespace

TEST_P(Positive, Test)
{
  // It's helpful to combine multiple calls of this function together, with stores collected
  // in-between, in hopes of triggering consensus match.
  // TODO(wonchanl): Also try keeping multiple stores alive at one time.
  const auto& [layout, unordered, read_only, use_tasks, use_inline] = GetParam();
  const auto& [dim, fortran]                                        = layout;
  test_body(dim, fortran, unordered, read_only, use_tasks, use_inline);
}

TEST_F(Attach, Negative)
{
  auto runtime = legate::Runtime::get_runtime();

  // Trying to detach a store without an attachment
  EXPECT_THROW(runtime->create_store(legate::Scalar{42}).detach(), std::invalid_argument);
  EXPECT_THROW(runtime->create_store(SHAPE_2D(), legate::int64()).detach(), std::invalid_argument);
  EXPECT_THROW(runtime->create_store(legate::int64()).detach(), std::invalid_argument);

  // Trying to attach to a NULL buffer
  EXPECT_THROW((void)runtime->create_store(SHAPE_2D(), legate::int64(), nullptr, true),
               std::invalid_argument);

  {
    // Trying to detach a sub-store
    auto mem = new std::int64_t[SHAPE_1D().volume()];
    auto l_store =
      runtime->create_store(SHAPE_1D(), legate::int64(), mem, /*read_only=*/true /*share*/);
    EXPECT_THROW(l_store.project(0, 1).detach(), std::invalid_argument);
    // We have to properly detach this
    l_store.detach();
    delete[] mem;
  }

  // Trying to attach a buffer smaller than what the store requires
  {
    std::vector<std::int64_t> test(3, 0);
    EXPECT_THROW(
      (void)runtime->create_store(SHAPE_1D(),
                                  legate::int64(),
                                  legate::ExternalAllocation::create_sysmem(
                                    test.data(), test.size() * sizeof(decltype(test)::value_type))),
      std::invalid_argument);
  }
}

}  // namespace attach
