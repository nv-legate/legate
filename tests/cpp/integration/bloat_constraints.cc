/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace bloat_constraints {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::int64_t BLOAT_TESTER = 0;

template <std::int32_t DIM>
struct BloatTester : public legate::LegateTask<BloatTester<DIM>> {
  static constexpr auto TASK_ID = legate::LocalTaskID{BLOAT_TESTER + DIM};

  static void cpu_variant(legate::TaskContext context)
  {
    auto source  = context.input(0);
    auto bloated = context.input(1);

    auto extents      = context.scalar(0).values<std::size_t>();
    auto low_offsets  = context.scalar(1).values<std::size_t>();
    auto high_offsets = context.scalar(2).values<std::size_t>();

    auto source_shape  = source.shape<DIM>();
    auto bloated_shape = bloated.shape<DIM>();

    if (source_shape.empty()) {
      return;
    }

    for (std::int32_t idx = 0; idx < DIM; ++idx) {
      auto low = std::max<std::int64_t>(
        0, source_shape.lo[idx] - static_cast<std::int64_t>(low_offsets[idx]));
      auto high =
        std::min<std::int64_t>(static_cast<std::int64_t>(extents[idx] - 1),
                               source_shape.hi[idx] + static_cast<std::int64_t>(high_offsets[idx]));
      EXPECT_EQ(low, bloated_shape.lo[idx]);
      EXPECT_EQ(high, bloated_shape.hi[idx]);
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_bloat_constraints";
  static void registration_callback(legate::Library library)
  {
    BloatTester<1>::register_variants(library);
    BloatTester<2>::register_variants(library);
    BloatTester<3>::register_variants(library);
  }
};

class BloatConstraint : public RegisterOnceFixture<Config> {};

struct BloatTestSpec {
  legate::tuple<std::uint64_t> extents;
  legate::tuple<std::uint64_t> low_offsets;
  legate::tuple<std::uint64_t> high_offsets;
};

void test_bloat(const BloatTestSpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto source  = runtime->create_store(spec.extents, legate::int64());
  auto bloated = runtime->create_store(spec.extents, legate::int64());
  runtime->issue_fill(source, legate::Scalar{std::int64_t{0}});
  runtime->issue_fill(bloated, legate::Scalar{std::int64_t{0}});

  auto task =
    runtime->create_task(context, static_cast<legate::LocalTaskID>(BLOAT_TESTER + source.dim()));
  auto part_source  = task.add_input(source);
  auto part_bloated = task.add_input(bloated);
  task.add_constraint(
    legate::bloat(part_source, part_bloated, spec.low_offsets, spec.high_offsets));
  task.add_scalar_arg(legate::Scalar{spec.extents.data()});
  task.add_scalar_arg(legate::Scalar{spec.low_offsets.data()});
  task.add_scalar_arg(legate::Scalar{spec.high_offsets.data()});

  runtime->submit(std::move(task));
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  {
    auto source  = runtime->create_store(legate::Shape{1, 2}, legate::float16());
    auto bloated = runtime->create_store(legate::Shape{2, 3, 4}, legate::int64());

    auto task = runtime->create_task(context, static_cast<legate::LocalTaskID>(BLOAT_TESTER + 2));
    auto part_source  = task.add_output(source);
    auto part_bloated = task.add_output(bloated);
    task.add_constraint(legate::bloat(part_source,
                                      part_bloated,
                                      legate::tuple<std::uint64_t>{2, 3},
                                      legate::tuple<std::uint64_t>{4, 5}));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }

  {
    auto source  = runtime->create_store(legate::Shape{1, 2}, legate::float16());
    auto bloated = runtime->create_store(legate::Shape{2, 3}, legate::int64());

    auto task = runtime->create_task(context, static_cast<legate::LocalTaskID>(BLOAT_TESTER + 2));
    auto part_source  = task.add_output(source);
    auto part_bloated = task.add_output(bloated);
    task.add_constraint(legate::bloat(part_source,
                                      part_bloated,
                                      legate::tuple<std::uint64_t>{2, 3, 3},
                                      legate::tuple<std::uint64_t>{4, 5}));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }

  {
    auto source  = runtime->create_store(legate::Shape{1, 2}, legate::float16());
    auto bloated = runtime->create_store(legate::Shape{2, 3}, legate::int64());

    auto task = runtime->create_task(context, static_cast<legate::LocalTaskID>(BLOAT_TESTER + 2));
    auto part_source  = task.add_output(source);
    auto part_bloated = task.add_output(bloated);
    task.add_constraint(legate::bloat(part_source,
                                      part_bloated,
                                      legate::tuple<std::uint64_t>{2, 3},
                                      legate::tuple<std::uint64_t>{4, 5, 3}));

    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }
}

}  // namespace

TEST_F(BloatConstraint, 1D)
{
  test_bloat({legate::tuple<std::uint64_t>{10},
              legate::tuple<std::uint64_t>{2},
              legate::tuple<std::uint64_t>{4}});
}

TEST_F(BloatConstraint, 2D)
{
  test_bloat({legate::tuple<std::uint64_t>{9, 9},
              legate::tuple<std::uint64_t>{2, 3},
              legate::tuple<std::uint64_t>{3, 4}});
}

TEST_F(BloatConstraint, 3D)
{
  test_bloat({legate::tuple<std::uint64_t>{10, 10, 10},
              legate::tuple<std::uint64_t>{2, 3, 4},
              legate::tuple<std::uint64_t>{4, 3, 2}});
}

TEST_F(BloatConstraint, Invalid) { test_invalid(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace bloat_constraints
