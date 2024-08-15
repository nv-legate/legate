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

#include "core/data/detail/logical_store.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace image_constraints {

// NOLINTBEGIN(readability-magic-numbers)

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_image_constraints";
  static void registration_callback(legate::Library library);
};

namespace {

constexpr std::int32_t TEST_MAX_DIM = 3;

[[nodiscard]] legate::Logger& logger()
{
  static legate::Logger log{std::string{Config::LIBRARY_NAME}};

  return log;
}

}  // namespace

enum TaskIDs : std::uint8_t {
  INIT_FUNC    = 0,
  IMAGE_TESTER = INIT_FUNC + TEST_MAX_DIM * 2,
};

template <std::int32_t DIM>
legate::Point<DIM> delinearize(std::size_t index,
                               std::size_t vol,
                               const legate::Point<DIM>& extents)
{
  legate::Point<DIM> result;
  for (std::int32_t dim = 0; dim < DIM; ++dim) {
    result[dim] = index * extents[dim] / vol;
  }
  return result;
}

template <std::int32_t DIM, bool RECT>
struct InitializeFunction : public legate::LegateTask<InitializeFunction<DIM, RECT>> {
  static constexpr std::int32_t TASK_ID =
    INIT_FUNC + static_cast<std::int32_t>(RECT) * TEST_MAX_DIM + DIM;

  struct InitializeRects {
    template <std::int32_t TGT_DIM>
    void operator()(legate::PhysicalStore& output,
                    const legate::Scalar& extents_scalar,
                    bool ascending)
    {
      auto shape   = output.shape<DIM>();
      auto extents = extents_scalar.value<legate::Point<TGT_DIM>>();
      auto acc     = output.write_accessor<legate::Rect<TGT_DIM>, DIM>();

      const auto vol = shape.volume();
      auto in_bounds = [&](const auto& point) {
        for (std::int32_t dim = 0; dim < TGT_DIM; ++dim) {
          if (point[dim] >= extents[dim]) {
            return false;
          }
        }
        return true;
      };
      const std::int64_t diff = ascending ? 1 : -1;
      auto idx                = static_cast<std::int64_t>(ascending ? 0 : vol - 1);

      for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
        auto lo = delinearize(idx, vol, extents);
        auto hi = lo + legate::Point<TGT_DIM>::ONES();

        if (in_bounds(hi)) {
          acc[*it] = legate::Rect<TGT_DIM>{lo, hi};
        } else {
          acc[*it] = legate::Rect<TGT_DIM>{lo, lo};
        }
        idx += diff;
      }
    }
  };

  struct InitializePoints {
    template <std::int32_t TGT_DIM>
    void operator()(legate::PhysicalStore& output,
                    const legate::Scalar& extents_scalar,
                    bool ascending)
    {
      auto shape   = output.shape<DIM>();
      auto extents = extents_scalar.value<legate::Point<TGT_DIM>>();
      auto acc     = output.write_accessor<legate::Point<TGT_DIM>, DIM>();

      const auto vol          = shape.volume();
      const std::int64_t diff = ascending ? 1 : -1;
      auto idx                = static_cast<std::int64_t>(ascending ? 0 : vol - 1);

      for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
        acc[*it] = delinearize(idx, vol, extents);
        idx += diff;
      }
    }
  };

  static void cpu_variant(legate::TaskContext context)
  {
    auto output    = context.output(0).data();
    auto extents   = context.scalar(0);
    auto ascending = context.scalar(1).value<bool>();
    if constexpr (RECT) {
      const auto& rect_type  = output.type().as_struct_type();
      const auto& point_type = rect_type.field_type(0).as_fixed_array_type();
      legate::dim_dispatch(
        point_type.num_elements(), InitializeRects{}, output, extents, ascending);
    } else {
      const auto& point_type = output.type().as_fixed_array_type();
      legate::dim_dispatch(
        point_type.num_elements(), InitializePoints{}, output, extents, ascending);
    }
  }

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context) { cpu_variant(context); }
#endif
};

template <std::int32_t DIM, bool RECT>
struct ImageTester : public legate::LegateTask<ImageTester<DIM, RECT>> {
  static constexpr std::int32_t TASK_ID =
    IMAGE_TESTER + static_cast<std::int32_t>(RECT) * TEST_MAX_DIM + DIM;

  struct CheckRects {
    template <std::int32_t TGT_DIM>
    void operator()(legate::PhysicalStore& func, const legate::Domain& range)
    {
      auto shape = func.shape<DIM>();
      if (shape.empty()) {
        return;
      }

      auto acc = func.read_accessor<legate::Rect<TGT_DIM>, DIM>();
      for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
        auto rect = acc[*it];
        for (legate::PointInRectIterator<TGT_DIM> rit(rect); rit.valid(); ++rit) {
          EXPECT_TRUE(range.contains(*rit));
        }
      }
    }
  };

  struct CheckPoints {
    template <std::int32_t TGT_DIM>
    void operator()(legate::PhysicalStore& func, const legate::Domain& range)
    {
      auto shape = func.shape<DIM>();
      if (shape.empty()) {
        return;
      }

      auto acc = func.read_accessor<legate::Point<TGT_DIM>, DIM>();
      for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
        auto p = acc[*it];
        EXPECT_TRUE(range.contains(p));
      }
    }
  };

  static void cpu_variant(legate::TaskContext context)
  {
    auto func  = context.input(0).data();
    auto range = context.input(1).domain();
    auto hint  = context.scalar(0).value<legate::ImageComputationHint>();

    switch (hint) {
      case legate::ImageComputationHint::NO_HINT: {
        EXPECT_TRUE(context.is_single_task() || range.get_volume() <= 1 || !range.dense());
        break;
      }
      case legate::ImageComputationHint::MIN_MAX:
      case legate::ImageComputationHint::FIRST_LAST: {
        EXPECT_TRUE(range.dense());
      }
    }

    if constexpr (RECT) {
      const auto& rect_type  = func.type().as_struct_type();
      const auto& point_type = rect_type.field_type(0).as_fixed_array_type();
      legate::dim_dispatch(point_type.num_elements(), CheckRects{}, func, range);
    } else {
      const auto& point_type = func.type().as_fixed_array_type();
      legate::dim_dispatch(point_type.num_elements(), CheckPoints{}, func, range);
    }

    if (context.get_task_index() == context.get_launch_domain().lo()) {
      logger().debug() << "== Image received in task 0 ==";
      for (legate::Domain::DomainPointIterator it(range); it; ++it) {
        logger().debug() << "  " << *it;
      }
      logger().debug() << "==============================";
    }
  }
};

/*static*/ void Config::registration_callback(legate::Library library)
{
  InitializeFunction<1, true>::register_variants(library);
  InitializeFunction<2, true>::register_variants(library);
  InitializeFunction<3, true>::register_variants(library);
  InitializeFunction<1, false>::register_variants(library);
  InitializeFunction<2, false>::register_variants(library);
  InitializeFunction<3, false>::register_variants(library);
  ImageTester<1, true>::register_variants(library);
  ImageTester<2, true>::register_variants(library);
  ImageTester<3, true>::register_variants(library);
  ImageTester<1, false>::register_variants(library);
  ImageTester<2, false>::register_variants(library);
  ImageTester<3, false>::register_variants(library);
}

class ImageConstraint : public RegisterOnceFixture<Config> {};

class Valid
  : public RegisterOnceFixture<Config>,
    public ::testing::WithParamInterface<std::tuple<legate::ImageComputationHint, bool, bool>> {};

INSTANTIATE_TEST_SUITE_P(
  ImageConstraint,
  Valid,
  ::testing::Combine(::testing::Values(legate::ImageComputationHint::NO_HINT,
                                       legate::ImageComputationHint::MIN_MAX,
                                       legate::ImageComputationHint::FIRST_LAST),
                     ::testing::Bool(),
                     ::testing::Bool()));

void initialize_function(const legate::LogicalStore& func,
                         const std::vector<std::uint64_t>& range_extents,
                         bool ascending)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto is_rect = func.type().code() == legate::Type::Code::STRUCT;
  auto task    = runtime->create_task(
    context,
    legate::LocalTaskID{static_cast<std::int64_t>(INIT_FUNC) +
                        static_cast<std::int32_t>(is_rect) * TEST_MAX_DIM + func.dim()});
  auto part = task.declare_partition();
  task.add_output(func, part);
  task.add_scalar_arg(legate::Scalar{range_extents});
  task.add_scalar_arg(legate::Scalar{ascending});
  task.add_constraint(legate::broadcast(part, legate::from_range(func.dim())));
  runtime->submit(std::move(task));

  func.impl()->reset_key_partition();
}

void check_image(const legate::LogicalStore& func,
                 const legate::LogicalStore& range,
                 legate::ImageComputationHint hint)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto is_rect = func.type().code() == legate::Type::Code::STRUCT;
  auto task    = runtime->create_task(
    context,
    legate::LocalTaskID{static_cast<std::int64_t>(IMAGE_TESTER) +
                        static_cast<std::int32_t>(is_rect) * TEST_MAX_DIM + func.dim()});
  auto part_domain = task.declare_partition();
  auto part_range  = task.declare_partition();

  task.add_input(func, part_domain);
  task.add_input(range, part_range);
  task.add_scalar_arg(legate::Scalar{legate::traits::detail::to_underlying(hint)});
  task.add_constraint(legate::image(part_domain, part_range, hint));

  runtime->submit(std::move(task));
}

struct ImageTestSpec {
  std::vector<std::uint64_t> domain_extents;
  std::vector<std::uint64_t> range_extents;
  legate::ImageComputationHint hint;
  bool is_rect;
  bool ascending;
};

void test_image(const ImageTestSpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  static_cast<void>(context);

  auto tgt_dim    = static_cast<std::int32_t>(spec.range_extents.size());
  auto image_type = spec.is_rect ? static_cast<legate::Type>(legate::rect_type(tgt_dim))
                                 : static_cast<legate::Type>(legate::point_type(tgt_dim));
  auto func       = runtime->create_store(legate::Shape{spec.domain_extents}, image_type);
  auto range      = runtime->create_store(legate::Shape{spec.range_extents}, legate::int64());

  initialize_function(func, spec.range_extents, spec.ascending);
  runtime->issue_fill(range, legate::Scalar(std::int64_t{1234}));
  check_image(func, range, spec.hint);
  runtime->issue_execution_fence();
  check_image(func, range, spec.hint);
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto create_task = [&](auto func, auto range) {
    auto task = runtime->create_task(
      context, legate::LocalTaskID{static_cast<std::int64_t>(IMAGE_TESTER) + func.dim()});
    auto part_domain = task.declare_partition();
    auto part_range  = task.declare_partition();

    task.add_input(func, part_domain);
    task.add_input(range, part_range);
    task.add_constraint(legate::image(part_domain, part_range));

    return task;
  };

  {
    auto func1  = runtime->create_store(legate::Shape{10, 10}, legate::int32());
    auto range1 = runtime->create_store(legate::Shape{10, 10}, legate::int64());
    auto task   = create_task(func1, range1);
    EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
  }

  {
    auto func2  = runtime->create_store(legate::Shape{4, 4}, legate::point_type(2));
    auto range2 = runtime->create_store(legate::Shape{10}, legate::int64());
    auto task   = create_task(func2, range2.promote(1, 1));
    EXPECT_THROW(runtime->submit(std::move(task)), std::runtime_error);
  }
}

TEST_P(Valid, 1D)
{
  auto& [hint, is_rect, ascending] = GetParam();
  test_image({{9}, {100}, hint, is_rect, ascending});
}

TEST_P(Valid, 2D)
{
  auto& [hint, is_rect, ascending] = GetParam();
  test_image({{4, 4}, {10, 10}, hint, is_rect, ascending});
}

TEST_P(Valid, 3D)
{
  auto& [hint, is_rect, ascending] = GetParam();
  test_image({{2, 3, 4}, {5, 5, 5}, hint, is_rect, ascending});
}

TEST_F(ImageConstraint, Invalid) { test_invalid(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace image_constraints
