/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

using ImageConstraint = DefaultFixture;

static const char* library_name = "test_image_constraints";

static const int32_t TEST_MAX_DIM = 3;

static legate::Logger logger(library_name);

enum TaskIDs {
  INIT_FUNC    = 0,
  IMAGE_TESTER = INIT_FUNC + TEST_MAX_DIM * 2,
};

template <int32_t DIM>
legate::Point<DIM> delinearize(size_t index, const legate::Point<DIM>& extents)
{
  legate::Point<DIM> result;
  for (int32_t dim = 0; dim < DIM; ++dim) {
    result[dim] = index % extents[dim];
    index       = index / extents[dim];
  }
  return result;
}

template <int32_t DIM, bool RECT>
struct InitializeFunction : public legate::LegateTask<InitializeFunction<DIM, RECT>> {
  struct InitializeRects {
    template <int32_t TGT_DIM>
    void operator()(legate::PhysicalStore& output,
                    const legate::Scalar& extents_scalar,
                    bool ascending)
    {
      auto shape   = output.shape<DIM>();
      auto extents = extents_scalar.value<legate::Point<TGT_DIM>>();
      auto acc     = output.write_accessor<legate::Rect<TGT_DIM>, DIM>();

      size_t vol     = shape.volume();
      size_t tgt_vol = 1;
      for (int32_t dim = 0; dim < TGT_DIM; ++dim) tgt_vol *= extents[dim];

      auto in_bounds = [&](const auto& point) {
        for (int32_t dim = 0; dim < TGT_DIM; ++dim)
          if (point[dim] >= extents[dim]) return false;
        return true;
      };
      int64_t idx  = ascending ? 0 : vol - 1;
      int64_t diff = ascending ? 1 : -1;
      for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
        auto lo = delinearize(idx * tgt_vol / vol, extents);
        auto hi = lo + legate::Point<TGT_DIM>::ONES();
        if (in_bounds(hi))
          acc[*it] = legate::Rect<TGT_DIM>(lo, hi);
        else
          acc[*it] = legate::Rect<TGT_DIM>(lo, lo);
        idx += diff;
      }
    }
  };

  struct InitializePoints {
    template <int32_t TGT_DIM>
    void operator()(legate::PhysicalStore& output,
                    const legate::Scalar& extents_scalar,
                    bool ascending)
    {
      auto shape   = output.shape<DIM>();
      auto extents = extents_scalar.value<legate::Point<TGT_DIM>>();
      auto acc     = output.write_accessor<legate::Point<TGT_DIM>, DIM>();

      size_t vol     = shape.volume();
      size_t tgt_vol = 1;
      for (int32_t dim = 0; dim < TGT_DIM; ++dim) tgt_vol *= extents[dim];

      int64_t idx  = ascending ? 0 : vol - 1;
      int64_t diff = ascending ? 1 : -1;
      for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
        auto p   = delinearize(idx * tgt_vol / vol, extents);
        acc[*it] = p;
        idx += diff;
      }
    }
  };

  static const int32_t TASK_ID = INIT_FUNC + static_cast<int32_t>(RECT) * TEST_MAX_DIM + DIM;
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
};

template <int32_t DIM, bool RECT>
struct ImageTester : public legate::LegateTask<ImageTester<DIM, RECT>> {
  static const int32_t TASK_ID = IMAGE_TESTER + static_cast<int32_t>(RECT) * TEST_MAX_DIM + DIM;
  struct CheckRects {
    template <int32_t TGT_DIM>
    void operator()(legate::PhysicalStore& func, const legate::Domain& range)
    {
      auto shape = func.shape<DIM>();
      if (shape.empty()) return;

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
    template <int32_t TGT_DIM>
    void operator()(legate::PhysicalStore& func, const legate::Domain& range)
    {
      auto shape = func.shape<DIM>();
      if (shape.empty()) return;

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
    EXPECT_TRUE(context.is_single_task() || range.get_volume() <= 1 || !range.dense());

    if constexpr (RECT) {
      const auto& rect_type  = func.type().as_struct_type();
      const auto& point_type = rect_type.field_type(0).as_fixed_array_type();
      legate::dim_dispatch(point_type.num_elements(), CheckRects{}, func, range);
    } else {
      const auto& point_type = func.type().as_fixed_array_type();
      legate::dim_dispatch(point_type.num_elements(), CheckPoints{}, func, range);
    }

    if (context.get_task_index() == context.get_launch_domain().lo()) {
      logger.debug() << "== Image received in task 0 ==";
      for (legate::Domain::DomainPointIterator it(range); it; ++it) {
        logger.debug() << "  " << *it;
      }
      logger.debug() << "==============================";
    }
  }
};

void prepare()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  InitializeFunction<1, true>::register_variants(context);
  InitializeFunction<2, true>::register_variants(context);
  InitializeFunction<3, true>::register_variants(context);
  InitializeFunction<1, false>::register_variants(context);
  InitializeFunction<2, false>::register_variants(context);
  InitializeFunction<3, false>::register_variants(context);
  ImageTester<1, true>::register_variants(context);
  ImageTester<2, true>::register_variants(context);
  ImageTester<3, true>::register_variants(context);
  ImageTester<1, false>::register_variants(context);
  ImageTester<2, false>::register_variants(context);
  ImageTester<3, false>::register_variants(context);
}

void initialize_function(legate::LogicalStore func,
                         const std::vector<size_t> range_extents,
                         bool ascending)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto is_rect = func.type().code() == legate::Type::Code::STRUCT;
  auto task    = runtime->create_task(
    context, INIT_FUNC + static_cast<int32_t>(is_rect) * TEST_MAX_DIM + func.dim());
  auto part = task.declare_partition();
  task.add_output(func, part);
  task.add_scalar_arg(legate::Scalar{range_extents});
  task.add_scalar_arg(legate::Scalar{ascending});
  task.add_constraint(legate::broadcast(part, legate::from_range(func.dim())));
  runtime->submit(std::move(task));

  func.impl()->reset_key_partition();
}

void check_image(legate::LogicalStore func, legate::LogicalStore range)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto is_rect = func.type().code() == legate::Type::Code::STRUCT;
  auto task    = runtime->create_task(
    context, IMAGE_TESTER + static_cast<int32_t>(is_rect) * TEST_MAX_DIM + func.dim());
  auto part_domain = task.declare_partition();
  auto part_range  = task.declare_partition();

  task.add_input(func, part_domain);
  task.add_input(range, part_range);
  task.add_constraint(legate::image(part_domain, part_range));

  runtime->submit(std::move(task));
}

struct ImageTestSpec {
  std::vector<size_t> domain_extents;
  std::vector<size_t> range_extents;
  bool is_rect;
};

void test_image(const ImageTestSpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  static_cast<void>(context);

  auto tgt_dim    = static_cast<std::int32_t>(spec.range_extents.size());
  auto image_type = spec.is_rect ? legate::rect_type(tgt_dim) : legate::point_type(tgt_dim);
  auto func  = runtime->create_store(legate::Shape{spec.domain_extents}, std::move(image_type));
  auto range = runtime->create_store(legate::Shape{spec.range_extents}, legate::int64());

  initialize_function(func, spec.range_extents, true);
  runtime->issue_fill(range, legate::Scalar(int64_t(1234)));
  check_image(func, range);
  runtime->issue_execution_fence();
  check_image(func, range);
  initialize_function(func, spec.range_extents, false);
  check_image(func, range);
  runtime->issue_execution_fence();
  check_image(func, range);
}

void test_invalid()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto func  = runtime->create_store(legate::Shape{10, 10}, legate::int32());
  auto range = runtime->create_store(legate::Shape{10, 10}, legate::int64());

  auto task        = runtime->create_task(context, IMAGE_TESTER + 1);
  auto part_domain = task.declare_partition();
  auto part_range  = task.declare_partition();

  task.add_input(func, part_domain);
  task.add_input(range, part_range);
  task.add_constraint(legate::image(part_domain, part_range));

  EXPECT_THROW(runtime->submit(std::move(task)), std::invalid_argument);
}

TEST_F(ImageConstraint, Point1D)
{
  prepare();
  test_image({{9}, {100}, false});
}

TEST_F(ImageConstraint, Point2D)
{
  prepare();
  test_image({{4, 4}, {10, 10}, false});
}

TEST_F(ImageConstraint, Point3D)
{
  prepare();
  test_image({{2, 3, 4}, {5, 5, 5}, false});
}

TEST_F(ImageConstraint, Rect1D)
{
  prepare();
  test_image({{9}, {100}, true});
}

TEST_F(ImageConstraint, Rect2D)
{
  prepare();
  test_image({{4, 4}, {10, 10}, true});
}

TEST_F(ImageConstraint, Rect3D)
{
  prepare();
  test_image({{2, 3, 4}, {5, 5, 5}, true});
}

TEST_F(ImageConstraint, Invalid)
{
  prepare();
  test_invalid();
}

}  // namespace image_constraints
