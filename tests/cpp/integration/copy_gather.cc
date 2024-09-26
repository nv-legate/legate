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

#include "copy_util.inl"
#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace copy_gather {

namespace {

constexpr std::int32_t CHECK_GATHER_TASK = FILL_INDIRECT_TASK + (TEST_MAX_DIM * TEST_MAX_DIM);

template <std::int32_t IND_DIM, std::int32_t SRC_DIM>
struct CheckGatherTask : public legate::LegateTask<CheckGatherTask<IND_DIM, SRC_DIM>> {
  struct CheckGatherTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::TaskContext context)
    {
      using VAL = legate::type_of_t<CODE>;

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
        ASSERT_EQ(copy, source);
      }
    }
  };

  static constexpr auto TASK_ID =
    legate::LocalTaskID{CHECK_GATHER_TASK + (IND_DIM * TEST_MAX_DIM) + SRC_DIM};

  static void cpu_variant(legate::TaskContext context)
  {
    auto type_code = context.input(0).type().code();
    type_dispatch_for_test(type_code, CheckGatherTaskBody{}, context);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_copy_gather";
  static void registration_callback(legate::Library library)
  {
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

    // CheckGatherTask<1, 1>::register_variants(library);
    CheckGatherTask<1, 2>::register_variants(library);
    // CheckGatherTask<1, 3>::register_variants(library);
    // CheckGatherTask<2, 1>::register_variants(library);
    CheckGatherTask<2, 2>::register_variants(library);
    CheckGatherTask<2, 3>::register_variants(library);
    CheckGatherTask<3, 1>::register_variants(library);
    CheckGatherTask<3, 2>::register_variants(library);
    // CheckGatherTask<3, 3>::register_variants(library);
  }
};

class GatherCopy : public RegisterOnceFixture<Config> {};

void check_gather_output(legate::Library library,
                         const legate::LogicalStore& src,
                         const legate::LogicalStore& tgt,
                         const legate::LogicalStore& ind)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  const auto task_id =
    legate::LocalTaskID{CHECK_GATHER_TASK + (ind.dim() * TEST_MAX_DIM) + src.dim()};
  auto task     = runtime->create_task(library, task_id);
  auto src_part = task.declare_partition();
  auto tgt_part = task.declare_partition();
  auto ind_part = task.declare_partition();
  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_input(ind, ind_part);

  task.add_constraint(legate::broadcast(src_part, legate::from_range(src.dim())));
  task.add_constraint(legate::broadcast(tgt_part, legate::from_range(tgt.dim())));
  task.add_constraint(legate::broadcast(ind_part, legate::from_range(ind.dim())));

  runtime->submit(std::move(task));
}

void test_gather(const std::vector<std::uint64_t>& ind_shape,
                 const std::vector<std::uint64_t>& src_shape,
                 const legate::Scalar& seed)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto type = seed.type();
  auto src  = runtime->create_store(legate::Shape{src_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{ind_shape}, type);
  auto ind  = runtime->create_store(legate::Shape{ind_shape}, legate::point_type(src_shape.size()));

  fill_input(library, src, seed);
  fill_indirect(library, ind, src);

  runtime->issue_gather(tgt, src, ind);

  check_gather_output(library, src, tgt, ind);
}

template <typename T>
void test_gather_reduction(const std::vector<std::uint64_t>& ind_shape,
                           const std::vector<std::uint64_t>& src_shape,
                           const legate::Scalar& seed,
                           T redop)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto type = seed.type();
  auto src  = runtime->create_store(legate::Shape{src_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{ind_shape}, type);
  auto ind  = runtime->create_store(legate::Shape{ind_shape}, legate::point_type(src_shape.size()));

  fill_input(library, src, seed);
  fill_indirect(library, ind, src);

  runtime->issue_gather(tgt, src, ind, redop);

  check_gather_output(library, src, tgt, ind);
}

}  // namespace

TEST_F(GatherCopy, 2Dto1D)
{
  const std::vector<std::uint64_t> ind_shape{5};
  const std::vector<std::uint64_t> src_shape{7, 11};
  const legate::Scalar seed{std::int64_t{123}};

  test_gather(ind_shape, src_shape, seed);
}

TEST_F(GatherCopy, 3Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{3, 7};
  const std::vector<std::uint64_t> src_shape{3, 2, 5};
  const legate::Scalar seed{std::uint32_t{456}};

  test_gather(ind_shape, src_shape, seed);
}

TEST_F(GatherCopy, 1Dto3D)
{
  const std::vector<std::uint64_t> ind_shape{2, 5, 4};
  const std::vector<std::uint64_t> src_shape{5};
  const legate::Scalar seed{789.0};

  test_gather(ind_shape, src_shape, seed);
}

TEST_F(GatherCopy, 2Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{4, 5};
  const std::vector<std::uint64_t> src_shape{10, 11};
  const legate::Scalar seed{std::int64_t{12}};

  test_gather(ind_shape, src_shape, seed);
}

TEST_F(GatherCopy, 2Dto3D)
{
  const std::vector<std::uint64_t> ind_shape{100, 100, 100};
  const std::vector<std::uint64_t> src_shape{10, 10};
  const legate::Scalar seed{7.0};

  test_gather(ind_shape, src_shape, seed);
}

TEST_F(GatherCopy, ReductionEnum2Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{0, 0};
  const std::vector<std::uint64_t> src_shape{7, 10};
  const legate::Scalar seed{2.0};
  constexpr legate::ReductionOpKind redop{legate::ReductionOpKind::ADD};

  test_gather_reduction(ind_shape, src_shape, seed, redop);
}

TEST_F(GatherCopy, ReductionInt322Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{0, 0};
  const std::vector<std::uint64_t> src_shape{7, 10};
  const legate::Scalar seed{2.0};
  constexpr std::int32_t redop{0};

  static_assert(redop == static_cast<std::int32_t>(legate::ReductionOpKind::ADD));
  static_assert(std::is_same_v<std::int32_t, std::underlying_type_t<legate::ReductionOpKind>>);

  test_gather_reduction(ind_shape, src_shape, seed, redop);
}

}  // namespace copy_gather
