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

namespace copy_gather {

// NOLINTBEGIN(readability-magic-numbers)

using Copy = DefaultFixture;

namespace {

constexpr const char library_name[]      = "test_copy_gather";
constexpr std::int32_t CHECK_GATHER_TASK = FILL_INDIRECT_TASK + TEST_MAX_DIM * TEST_MAX_DIM;

}  // namespace

template <std::int32_t IND_DIM, std::int32_t SRC_DIM>
struct CheckGatherTask : public legate::LegateTask<CheckGatherTask<IND_DIM, SRC_DIM>> {
  struct CheckGatherTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::TaskContext context)
    {
      using VAL = legate::type_of<CODE>;

      auto src_store = context.input(0).data();
      auto tgt_store = context.input(1).data();
      auto ind_store = context.input(2).data();

      auto ind_shape = ind_store.shape<IND_DIM>();
      if (ind_shape.empty()) {
        return;
      }

      auto src_acc = src_store.read_accessor<VAL, SRC_DIM>();
      auto tgt_acc = tgt_store.read_accessor<VAL, IND_DIM>();
      auto ind_acc = ind_store.read_accessor<legate::Point<SRC_DIM>, IND_DIM>();

      for (legate::PointInRectIterator<IND_DIM> it(ind_shape); it.valid(); ++it) {
        auto copy   = tgt_acc[*it];
        auto source = src_acc[ind_acc[*it]];
        EXPECT_EQ(copy, source);
      }
    }
  };

  static constexpr std::int32_t TASK_ID = CHECK_GATHER_TASK + IND_DIM * TEST_MAX_DIM + SRC_DIM;

  static void cpu_variant(legate::TaskContext context)
  {
    auto type_code = context.input(0).type().code();
    type_dispatch_for_test(type_code, CheckGatherTaskBody{}, context);
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
  auto context = runtime->create_library(library_name);
  FillTask<1>::register_variants(context);
  FillTask<2>::register_variants(context);
  FillTask<3>::register_variants(context);

  // XXX: Tasks unused by the test cases are commented out
  // FillIndirectTask<1, 1>::register_variants(context);
  FillIndirectTask<1, 2>::register_variants(context);
  // FillIndirectTask<1, 3>::register_variants(context);
  // FillIndirectTask<2, 1>::register_variants(context);
  FillIndirectTask<2, 2>::register_variants(context);
  FillIndirectTask<2, 3>::register_variants(context);
  FillIndirectTask<3, 1>::register_variants(context);
  FillIndirectTask<3, 2>::register_variants(context);
  // FillIndirectTask<3, 3>::register_variants(context);

  // CheckGatherTask<1, 1>::register_variants(context);
  CheckGatherTask<1, 2>::register_variants(context);
  // CheckGatherTask<1, 3>::register_variants(context);
  // CheckGatherTask<2, 1>::register_variants(context);
  CheckGatherTask<2, 2>::register_variants(context);
  CheckGatherTask<2, 3>::register_variants(context);
  CheckGatherTask<3, 1>::register_variants(context);
  CheckGatherTask<3, 2>::register_variants(context);
  // CheckGatherTask<3, 3>::register_variants(context);
}

void check_gather_output(legate::Library library,
                         const legate::LogicalStore& src,
                         const legate::LogicalStore& tgt,
                         const legate::LogicalStore& ind)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto machine       = runtime->get_machine();
  const auto task_id = CHECK_GATHER_TASK + ind.dim() * TEST_MAX_DIM + src.dim();
  auto task          = runtime->create_task(library, task_id);
  auto src_part      = task.declare_partition();
  auto tgt_part      = task.declare_partition();
  auto ind_part      = task.declare_partition();
  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_input(ind, ind_part);

  task.add_constraint(legate::broadcast(src_part, legate::from_range(src.dim())));
  task.add_constraint(legate::broadcast(tgt_part, legate::from_range(tgt.dim())));
  task.add_constraint(legate::broadcast(ind_part, legate::from_range(ind.dim())));

  runtime->submit(std::move(task));
}

struct GatherSpec {
  std::vector<std::uint64_t> ind_shape;
  std::vector<std::uint64_t> data_shape;
  legate::Scalar seed;

  [[nodiscard]] std::string to_string() const
  {
    std::stringstream ss;
    ss << "source shape: " << ::to_string(data_shape)
       << ", indirection/target shape: " << ::to_string(ind_shape)
       << ", data type: " << seed.type().to_string();
    return std::move(ss).str();
  }
};

void test_gather(const GatherSpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  auto type = spec.seed.type();
  auto src  = runtime->create_store(legate::Shape{spec.data_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{spec.ind_shape}, type);
  auto ind  = runtime->create_store(legate::Shape{spec.ind_shape},
                                   legate::point_type(spec.data_shape.size()));

  fill_input(library, src, spec.seed);
  fill_indirect(library, ind, src);

  runtime->issue_gather(tgt, src, ind);

  check_gather_output(library, src, tgt, ind);
}

TEST_F(Copy, Gather2Dto1D)
{
  register_tasks();
  const std::vector<std::uint64_t> shape1d{5};
  test_gather(GatherSpec{shape1d, {7, 11}, legate::Scalar{std::int64_t{123}}});
}

TEST_F(Copy, Gather3Dto2D)
{
  register_tasks();
  test_gather(GatherSpec{{3, 7}, {3, 2, 5}, legate::Scalar{std::uint32_t{456}}});
}

TEST_F(Copy, Gather1Dto3D)
{
  register_tasks();
  const std::vector<std::uint64_t> shape1d{5};
  test_gather(GatherSpec{{2, 5, 4}, shape1d, legate::Scalar{789.0}});
}

TEST_F(Copy, Gather2Dto2D)
{
  register_tasks();
  test_gather(GatherSpec{{4, 5}, {10, 11}, legate::Scalar{std::int64_t{12}}});
}

TEST_F(Copy, Gather2Dto3D)
{
  register_tasks();
  test_gather(GatherSpec{{100, 100, 100}, {10, 10}, legate::Scalar{7.0}});
}

// NOLINTEND(readability-magic-numbers)

}  // namespace copy_gather
