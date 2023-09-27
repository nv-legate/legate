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

#include <gtest/gtest.h>

#include "copy_util.inl"
#include "legate.h"

namespace copy_gather_scatter {

static const char* library_name = "test_copy_gather_scatter";
static legate::Logger logger(library_name);

constexpr int32_t CHECK_GATHER_SCATTER_TASK = FILL_INDIRECT_TASK + TEST_MAX_DIM * TEST_MAX_DIM;

template <int32_t SRC_DIM, int32_t IND_DIM, int32_t TGT_DIM>
struct CheckGatherScatterTask
  : public legate::LegateTask<CheckGatherScatterTask<SRC_DIM, IND_DIM, TGT_DIM>> {
  struct CheckGatherScatterTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::TaskContext context)
    {
      using VAL = legate::legate_type_of<CODE>;

      auto src_store     = context.input(0).data();
      auto tgt_store     = context.input(1).data();
      auto src_ind_store = context.input(2).data();
      auto tgt_ind_store = context.input(3).data();
      auto init          = context.scalar(0).value<VAL>();

      auto src_shape = src_store.shape<SRC_DIM>();
      static_cast<void>(src_shape);
      auto tgt_shape = tgt_store.shape<TGT_DIM>();
      auto ind_shape = src_ind_store.shape<IND_DIM>();

      legate::Buffer<bool, TGT_DIM> mask(tgt_shape, legate::Memory::Kind::SYSTEM_MEM);
      for (legate::PointInRectIterator<TGT_DIM> it(tgt_shape); it.valid(); ++it) mask[*it] = false;

      auto src_acc     = src_store.read_accessor<VAL, SRC_DIM>();
      auto tgt_acc     = tgt_store.read_accessor<VAL, TGT_DIM>();
      auto src_ind_acc = src_ind_store.read_accessor<legate::Point<SRC_DIM>, IND_DIM>();
      auto tgt_ind_acc = tgt_ind_store.read_accessor<legate::Point<TGT_DIM>, IND_DIM>();

      for (legate::PointInRectIterator<IND_DIM> it(ind_shape); it.valid(); ++it) {
        auto src_point = src_ind_acc[*it];
        auto tgt_point = tgt_ind_acc[*it];
        auto source    = src_acc[src_point];
        auto copy      = tgt_acc[tgt_point];
        EXPECT_EQ(copy, source);
        mask[tgt_point] = true;
      }

      for (legate::PointInRectIterator<TGT_DIM> it(tgt_shape); it.valid(); ++it) {
        auto p = *it;
        if (mask[p]) continue;
        EXPECT_EQ(tgt_acc[p], init);
      }
    }
  };

  static const int32_t TASK_ID = CHECK_GATHER_SCATTER_TASK + SRC_DIM * TEST_MAX_DIM * TEST_MAX_DIM +
                                 IND_DIM * TEST_MAX_DIM + TGT_DIM;
  static void cpu_variant(legate::TaskContext context)
  {
    auto type_code = context.input(0).type().code();
    type_dispatch_for_test(type_code, CheckGatherScatterTaskBody{}, context);
  }
};

struct GatherScatterSpec {
  std::vector<size_t> src_shape;
  std::vector<size_t> ind_shape;
  std::vector<size_t> tgt_shape;
  legate::Scalar seed;
  legate::Scalar init;

  std::string to_string() const
  {
    std::stringstream ss;
    ss << "source shape: " << ::to_string(src_shape)
       << ", indirection shape: " << ::to_string(ind_shape)
       << ", target shape: " << ::to_string(tgt_shape)
       << ", data type: " << seed.type().to_string();
    return std::move(ss).str();
  }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  FillTask<1>::register_variants(library);
  FillTask<2>::register_variants(library);
  FillTask<3>::register_variants(library);

  FillIndirectTask<1, 1>::register_variants(library);
  FillIndirectTask<1, 2>::register_variants(library);
  FillIndirectTask<1, 3>::register_variants(library);
  FillIndirectTask<2, 1>::register_variants(library);
  FillIndirectTask<2, 2>::register_variants(library);
  FillIndirectTask<2, 3>::register_variants(library);
  FillIndirectTask<3, 1>::register_variants(library);
  FillIndirectTask<3, 2>::register_variants(library);
  FillIndirectTask<3, 3>::register_variants(library);

  CheckGatherScatterTask<1, 2, 3>::register_variants(library);
  CheckGatherScatterTask<2, 3, 1>::register_variants(library);
  CheckGatherScatterTask<3, 1, 2>::register_variants(library);
  CheckGatherScatterTask<3, 3, 3>::register_variants(library);
  CheckGatherScatterTask<2, 2, 3>::register_variants(library);
}

void check_gather_scatter_output(legate::Library library,
                                 const legate::LogicalStore& src,
                                 const legate::LogicalStore& tgt,
                                 const legate::LogicalStore& src_ind,
                                 const legate::LogicalStore& tgt_ind,
                                 const legate::Scalar& init)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();

  int32_t task_id = CHECK_GATHER_SCATTER_TASK + src.dim() * TEST_MAX_DIM * TEST_MAX_DIM +
                    src_ind.dim() * TEST_MAX_DIM + tgt.dim();

  auto task = runtime->create_task(library, task_id);

  auto src_part     = task.declare_partition();
  auto tgt_part     = task.declare_partition();
  auto src_ind_part = task.declare_partition();
  auto tgt_ind_part = task.declare_partition();
  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_input(src_ind, src_ind_part);
  task.add_input(tgt_ind, tgt_ind_part);
  task.add_scalar_arg(init);

  task.add_constraint(legate::broadcast(src_part, legate::from_range<int32_t>(src.dim())));
  task.add_constraint(legate::broadcast(tgt_part, legate::from_range<int32_t>(tgt.dim())));
  task.add_constraint(legate::broadcast(src_ind_part, legate::from_range<int32_t>(src_ind.dim())));
  task.add_constraint(legate::broadcast(tgt_ind_part, legate::from_range<int32_t>(tgt_ind.dim())));

  runtime->submit(std::move(task));
}

void test_gather_scatter(const GatherScatterSpec& spec)
{
  assert(spec.seed.type() == spec.init.type());
  logger.print() << "Gather-scatter Copy: " << spec.to_string();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  auto type    = spec.seed.type();
  auto src     = runtime->create_store(spec.src_shape, type);
  auto tgt     = runtime->create_store(spec.tgt_shape, type);
  auto src_ind = runtime->create_store(spec.ind_shape, legate::point_type(spec.src_shape.size()));
  auto tgt_ind = runtime->create_store(spec.ind_shape, legate::point_type(spec.tgt_shape.size()));

  fill_input(library, src, spec.seed);
  fill_indirect(library, src_ind, src);
  fill_indirect(library, tgt_ind, tgt);
  runtime->issue_fill(tgt, spec.init);
  runtime->issue_scatter_gather(tgt, tgt_ind, src, src_ind);

  check_gather_scatter_output(library, src, tgt, src_ind, tgt_ind, spec.init);
}

TEST(Copy, GatherScatter1Dto3Dvia2D)
{
  legate::Core::perform_registration<register_tasks>();
  std::vector<size_t> shape1d{5};
  test_gather_scatter(GatherScatterSpec{
    shape1d, {7, 11}, {10, 10, 10}, legate::Scalar(int64_t(123)), legate::Scalar(int64_t(42))});
}

TEST(Copy, GatherScatter2Dto1Dvia3D)
{
  legate::Core::perform_registration<register_tasks>();
  std::vector<size_t> shape1d{1000};
  test_gather_scatter(GatherScatterSpec{
    {3, 7}, {3, 6, 5}, shape1d, legate::Scalar(uint32_t(456)), legate::Scalar(uint32_t(42))});
}

TEST(Copy, GatherScatter3Dto2Dvia1D)
{
  legate::Core::perform_registration<register_tasks>();
  std::vector<size_t> shape1d{100};
  test_gather_scatter(GatherScatterSpec{
    {4, 5, 2}, shape1d, {50, 50}, legate::Scalar(int64_t(12)), legate::Scalar(int64_t(42))});
}

TEST(Copy, GatherScatter3Dto3Dvia3D)
{
  legate::Core::perform_registration<register_tasks>();
  test_gather_scatter(GatherScatterSpec{{10, 10, 10},
                                        {5, 4, 2},
                                        {10, 10, 10},
                                        legate::Scalar(int64_t(1)),
                                        legate::Scalar(int64_t(42))});
}

TEST(Copy, GatherScatter2Dto3Dvia2D)
{
  legate::Core::perform_registration<register_tasks>();
  test_gather_scatter(GatherScatterSpec{
    {27, 33}, {11, 7}, {132, 121, 3}, legate::Scalar(int64_t(2)), legate::Scalar(int64_t(84))});
}

}  // namespace copy_gather_scatter
