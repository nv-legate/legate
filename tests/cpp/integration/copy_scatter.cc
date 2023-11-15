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

#include "copy_util.inl"
#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace copy_scatter {

using Copy = DefaultFixture;

static const char* library_name = "test_copy_scatter";
static legate::Logger logger(library_name);

constexpr int32_t CHECK_SCATTER_TASK = FILL_INDIRECT_TASK + TEST_MAX_DIM * TEST_MAX_DIM;

template <int32_t IND_DIM, int32_t TGT_DIM>
struct CheckScatterTask : public legate::LegateTask<CheckScatterTask<IND_DIM, TGT_DIM>> {
  struct CheckScatterTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::TaskContext context)
    {
      using VAL = legate::type_of<CODE>;

      auto src_store = context.input(0).data();
      auto tgt_store = context.input(1).data();
      auto ind_store = context.input(2).data();
      auto init      = context.scalar(0).value<VAL>();

      auto ind_shape = ind_store.shape<IND_DIM>();
      if (ind_shape.empty()) {
        return;
      }

      auto tgt_shape = tgt_store.shape<TGT_DIM>();

      legate::Buffer<bool, TGT_DIM> mask(tgt_shape, legate::Memory::Kind::SYSTEM_MEM);
      for (legate::PointInRectIterator<TGT_DIM> it(tgt_shape); it.valid(); ++it) {
        mask[*it] = false;
      }

      auto src_acc = src_store.read_accessor<VAL, IND_DIM>();
      auto tgt_acc = tgt_store.read_accessor<VAL, TGT_DIM>();
      auto ind_acc = ind_store.read_accessor<legate::Point<TGT_DIM>, IND_DIM>();

      for (legate::PointInRectIterator<IND_DIM> it(ind_shape); it.valid(); ++it) {
        auto p      = ind_acc[*it];
        auto copy   = tgt_acc[p];
        auto source = src_acc[*it];
        EXPECT_EQ(copy, source);
        mask[p] = true;
      }

      for (legate::PointInRectIterator<TGT_DIM> it(tgt_shape); it.valid(); ++it) {
        auto p = *it;
        if (mask[p]) {
          continue;
        }
        EXPECT_EQ(tgt_acc[p], init);
      }
    }
  };

  static const int32_t TASK_ID = CHECK_SCATTER_TASK + IND_DIM * TEST_MAX_DIM + TGT_DIM;
  static void cpu_variant(legate::TaskContext context)
  {
    auto type_code = context.input(0).type().code();
    type_dispatch_for_test(type_code, CheckScatterTaskBody{}, context);
  }
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  FillTask<1>::register_variants(library);
  FillTask<2>::register_variants(library);
  FillTask<3>::register_variants(library);

  // XXX: Tasks unused by the test cases are commented out
  // FillIndirectTask<1, 1>::register_variants(library);
  FillIndirectTask<1, 2>::register_variants(library);
  // FillIndirectTask<1, 3>::register_variants(library);
  // FillIndirectTask<2, 1>::register_variants(library);
  FillIndirectTask<2, 2>::register_variants(library);
  FillIndirectTask<2, 3>::register_variants(library);
  FillIndirectTask<3, 1>::register_variants(library);
  FillIndirectTask<3, 2>::register_variants(library);
  // FillIndirectTask<3, 3>::register_variants(library);

  // CheckScatterTask<1, 1>::register_variants(library);
  CheckScatterTask<1, 2>::register_variants(library);
  // CheckScatterTask<1, 3>::register_variants(library);
  // CheckScatterTask<2, 1>::register_variants(library);
  CheckScatterTask<2, 2>::register_variants(library);
  CheckScatterTask<2, 3>::register_variants(library);
  CheckScatterTask<3, 1>::register_variants(library);
  CheckScatterTask<3, 2>::register_variants(library);
  // CheckScatterTask<3, 3>::register_variants(library);
}

struct ScatterSpec {
  std::vector<size_t> ind_shape;
  std::vector<size_t> data_shape;
  legate::Scalar seed;
  legate::Scalar init;

  std::string to_string() const
  {
    std::stringstream ss;

    ss << "indirection/source shape: " << ::to_string(ind_shape)
       << ", target shape: " << ::to_string(data_shape);
    ss << ", data type: " << seed.type().to_string();
    return std::move(ss).str();
  }
};

void check_scatter_output(legate::Library library,
                          const legate::LogicalStore& src,
                          const legate::LogicalStore& tgt,
                          const legate::LogicalStore& ind,
                          const legate::Scalar& init)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();

  int32_t task_id = CHECK_SCATTER_TASK + ind.dim() * TEST_MAX_DIM + tgt.dim();

  auto task = runtime->create_task(library, task_id);

  auto src_part = task.declare_partition();
  auto tgt_part = task.declare_partition();
  auto ind_part = task.declare_partition();
  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_input(ind, ind_part);
  task.add_scalar_arg(init);

  task.add_constraint(legate::broadcast(src_part, legate::from_range<int32_t>(src.dim())));
  task.add_constraint(legate::broadcast(tgt_part, legate::from_range<int32_t>(tgt.dim())));
  task.add_constraint(legate::broadcast(ind_part, legate::from_range<int32_t>(ind.dim())));

  runtime->submit(std::move(task));
}

void test_scatter(const ScatterSpec& spec)
{
  assert(spec.seed.type() == spec.init.type());
  logger.print() << "Scatter Copy: " << spec.to_string();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  auto type = spec.seed.type();
  auto src  = runtime->create_store(legate::Shape{spec.ind_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{spec.data_shape}, type);
  auto ind  = runtime->create_store(legate::Shape{spec.ind_shape},
                                   legate::point_type(spec.data_shape.size()));

  fill_input(library, src, spec.seed);
  fill_indirect(library, ind, tgt);
  runtime->issue_fill(tgt, spec.init);
  runtime->issue_scatter(tgt, ind, src);

  check_scatter_output(library, src, tgt, ind, spec.init);
}

// Note that the volume of indirection field should be smaller than that of the target to avoid
// duplicate updates on the same element, whose semantics is undefined.
TEST_F(Copy, Scatter1Dto2D)
{
  register_tasks();
  std::vector<size_t> shape1d{5};
  test_scatter(
    ScatterSpec{shape1d, {7, 11}, legate::Scalar(int64_t(123)), legate::Scalar(int64_t(42))});
}

TEST_F(Copy, Scatter2Dto3D)
{
  register_tasks();
  test_scatter(
    ScatterSpec{{3, 7}, {3, 6, 5}, legate::Scalar(uint32_t(456)), legate::Scalar(uint32_t(42))});
}

TEST_F(Copy, Scatter2Dto2D)
{
  register_tasks();
  test_scatter(
    ScatterSpec{{4, 5}, {10, 11}, legate::Scalar(int64_t(12)), legate::Scalar(int64_t(42))});
}

TEST_F(Copy, Scatter3Dto2D)
{
  register_tasks();
  test_scatter(
    ScatterSpec{{10, 10, 10}, {200, 200}, legate::Scalar(int64_t(1)), legate::Scalar(int64_t(42))});
}

}  // namespace copy_scatter
