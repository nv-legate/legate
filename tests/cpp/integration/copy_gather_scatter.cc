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

namespace copy_gather_scatter {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::int32_t CHECK_GATHER_SCATTER_TASK = FILL_INDIRECT_TASK + TEST_MAX_DIM * TEST_MAX_DIM;

}  // namespace

template <std::int32_t SRC_DIM, std::int32_t IND_DIM, std::int32_t TGT_DIM>
struct CheckGatherScatterTask
  : public legate::LegateTask<CheckGatherScatterTask<SRC_DIM, IND_DIM, TGT_DIM>> {
  struct CheckGatherScatterTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::TaskContext context)
    {
      using VAL = legate::type_of_t<CODE>;

      auto src_store     = context.input(0).data();
      auto tgt_store     = context.input(1).data();
      auto src_ind_store = context.input(2).data();
      auto tgt_ind_store = context.input(3).data();
      auto init          = context.scalar(0).value<VAL>();

      auto src_shape = src_store.shape<SRC_DIM>();
      static_cast<void>(src_shape);
      auto tgt_shape = tgt_store.shape<TGT_DIM>();
      auto ind_shape = src_ind_store.shape<IND_DIM>();
      if (ind_shape.empty()) {
        return;
      }

      const legate::Buffer<bool, TGT_DIM> mask{tgt_shape, legate::Memory::Kind::SYSTEM_MEM};
      for (legate::PointInRectIterator<TGT_DIM> it{tgt_shape}; it.valid(); ++it) {
        mask[*it] = false;
      }

      auto src_acc     = src_store.read_accessor<VAL, SRC_DIM>();
      auto tgt_acc     = tgt_store.read_accessor<VAL, TGT_DIM>();
      auto src_ind_acc = src_ind_store.read_accessor<legate::Point<SRC_DIM>, IND_DIM>();
      auto tgt_ind_acc = tgt_ind_store.read_accessor<legate::Point<TGT_DIM>, IND_DIM>();

      for (legate::PointInRectIterator<IND_DIM> it{ind_shape}; it.valid(); ++it) {
        auto src_point = src_ind_acc[*it];
        auto tgt_point = tgt_ind_acc[*it];
        auto source    = src_acc[src_point];
        auto copy      = tgt_acc[tgt_point];
        EXPECT_EQ(copy, source);
        mask[tgt_point] = true;
      }

      for (legate::PointInRectIterator<TGT_DIM> it{tgt_shape}; it.valid(); ++it) {
        auto p = *it;
        if (mask[p]) {
          continue;
        }
        EXPECT_EQ(tgt_acc[p], init);
      }
    }
  };

  static constexpr std::int32_t TASK_ID = CHECK_GATHER_SCATTER_TASK +
                                          SRC_DIM * TEST_MAX_DIM * TEST_MAX_DIM +
                                          IND_DIM * TEST_MAX_DIM + TGT_DIM;

  static void cpu_variant(legate::TaskContext context)
  {
    auto type_code = context.input(0).type().code();
    type_dispatch_for_test(type_code, CheckGatherScatterTaskBody{}, context);
  }
};

struct GatherScatterSpec {
  std::vector<std::uint64_t> src_shape;
  std::vector<std::uint64_t> ind_shape;
  std::vector<std::uint64_t> tgt_shape;
  legate::Scalar seed;
  legate::Scalar init;

  [[nodiscard]] std::string to_string() const
  {
    std::stringstream ss;
    ss << "source shape: " << ::to_string(src_shape)
       << ", indirection shape: " << ::to_string(ind_shape)
       << ", target shape: " << ::to_string(tgt_shape)
       << ", data type: " << seed.type().to_string();
    return std::move(ss).str();
  }
};

template <typename T>
struct GatherScatterReductionSpec : GatherScatterSpec {
  T redop;
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_copy_gather_scatter";
  static void registration_callback(legate::Library library)
  {
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
    CheckGatherScatterTask<2, 2, 2>::register_variants(library);
    CheckGatherScatterTask<2, 2, 3>::register_variants(library);
  }
};

class ScatterGatherCopy : public RegisterOnceFixture<Config> {};

void check_gather_scatter_output(legate::Library library,
                                 const legate::LogicalStore& src,
                                 const legate::LogicalStore& tgt,
                                 const legate::LogicalStore& src_ind,
                                 const legate::LogicalStore& tgt_ind,
                                 const legate::Scalar& init)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();

  const auto task_id = static_cast<legate::LocalTaskID>(CHECK_GATHER_SCATTER_TASK +
                                                        src.dim() * TEST_MAX_DIM * TEST_MAX_DIM +
                                                        src_ind.dim() * TEST_MAX_DIM + tgt.dim());

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

  task.add_constraint(legate::broadcast(src_part, legate::from_range(src.dim())));
  task.add_constraint(legate::broadcast(tgt_part, legate::from_range(tgt.dim())));
  task.add_constraint(legate::broadcast(src_ind_part, legate::from_range(src_ind.dim())));
  task.add_constraint(legate::broadcast(tgt_ind_part, legate::from_range(tgt_ind.dim())));

  runtime->submit(std::move(task));
}

template <typename T>
void test_gather_scatter_impl(const GatherScatterSpec& spec, std::optional<T> redop = std::nullopt)
{
  LEGATE_ASSERT(spec.seed.type() == spec.init.type());

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto type = spec.seed.type();
  auto src  = runtime->create_store(legate::Shape{spec.src_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{spec.tgt_shape}, type);
  auto src_ind =
    runtime->create_store(legate::Shape{spec.ind_shape}, legate::point_type(spec.src_shape.size()));
  auto tgt_ind =
    runtime->create_store(legate::Shape{spec.ind_shape}, legate::point_type(spec.tgt_shape.size()));

  fill_input(library, src, spec.seed);
  fill_indirect(library, src_ind, src);
  fill_indirect(library, tgt_ind, tgt);
  runtime->issue_fill(tgt, spec.init);
  if (redop == std::nullopt) {
    runtime->issue_scatter_gather(tgt, tgt_ind, src, src_ind);
  } else {
    runtime->issue_scatter_gather(tgt, tgt_ind, src, src_ind, redop);
  }

  check_gather_scatter_output(library, src, tgt, src_ind, tgt_ind, spec.init);
}

void test_gather_scatter(const GatherScatterSpec& spec)
{
  test_gather_scatter_impl<std::int32_t>(spec);
}

void test_gather_scatter_reduction(const GatherScatterReductionSpec<legate::ReductionOpKind>& spec)
{
  test_gather_scatter_impl<legate::ReductionOpKind>(spec, spec.redop);
}

void test_gather_scatter_reduction_int32(const GatherScatterReductionSpec<std::int32_t>& spec)
{
  test_gather_scatter_impl<std::int32_t>(spec, spec.redop);
}

TEST_F(ScatterGatherCopy, 1Dto3Dvia2D)
{
  const std::vector<std::uint64_t> shape1d{5};
  test_gather_scatter(GatherScatterSpec{shape1d,
                                        {7, 11},
                                        {10, 10, 10},
                                        legate::Scalar{std::int64_t{123}},
                                        legate::Scalar{std::int64_t{42}}});
}

TEST_F(ScatterGatherCopy, 2Dto1Dvia3D)
{
  const std::vector<std::uint64_t> shape1d{1000};
  test_gather_scatter(GatherScatterSpec{{3, 7},
                                        {3, 6, 5},
                                        shape1d,
                                        legate::Scalar{std::uint32_t{456}},
                                        legate::Scalar{std::uint32_t{42}}});
}

TEST_F(ScatterGatherCopy, 3Dto2Dvia1D)
{
  const std::vector<std::uint64_t> shape1d{100};
  test_gather_scatter(GatherScatterSpec{{4, 5, 2},
                                        shape1d,
                                        {50, 50},
                                        legate::Scalar{std::int64_t{12}},
                                        legate::Scalar{std::int64_t{42}}});
}

TEST_F(ScatterGatherCopy, 3Dto3Dvia3D)
{
  test_gather_scatter(GatherScatterSpec{{10, 10, 10},
                                        {5, 4, 2},
                                        {10, 10, 10},
                                        legate::Scalar{std::int64_t{1}},
                                        legate::Scalar{std::int64_t{42}}});
}

TEST_F(ScatterGatherCopy, 2Dto3Dvia2D)
{
  test_gather_scatter(GatherScatterSpec{{27, 33},
                                        {11, 7},
                                        {132, 121, 3},
                                        legate::Scalar{std::int64_t{2}},
                                        legate::Scalar{std::int64_t{84}}});
}

TEST_F(ScatterGatherCopy, ReductionEnum2Dto2Dvia2D)
{
  const std::vector<std::uint64_t> src_shape{10, 10};
  const std::vector<std::uint64_t> ind_shape{0, 0};
  const std::vector<std::uint64_t> tgt_shape{10, 10};
  const legate::Scalar seed{std::int64_t{12}};
  const legate::Scalar init{std::int64_t{42}};
  const legate::ReductionOpKind redop{legate::ReductionOpKind::ADD};

  // Test with redop as ReductionOpKind
  test_gather_scatter_reduction(GatherScatterReductionSpec<legate::ReductionOpKind>{
    {src_shape, ind_shape, tgt_shape, seed, init}, redop});
}

TEST_F(ScatterGatherCopy, ReductionInt322Dto2Dvia2D)
{
  const std::vector<std::uint64_t> src_shape{10, 10};
  const std::vector<std::uint64_t> ind_shape{0, 0};
  const std::vector<std::uint64_t> tgt_shape{10, 10};
  const legate::Scalar seed{std::int64_t{12}};
  const legate::Scalar init{std::int64_t{42}};
  // ReductionOpKind::ADD
  const std::int32_t redop{0};

  static_assert(redop == static_cast<std::int32_t>(legate::ReductionOpKind::ADD));
  static_assert(std::is_same_v<std::int32_t, std::underlying_type_t<legate::ReductionOpKind>>);

  // Test with redop as int32
  test_gather_scatter_reduction_int32(
    GatherScatterReductionSpec<std::int32_t>{{src_shape, ind_shape, tgt_shape, seed, init}, redop});
}

// NOLINTEND(readability-magic-numbers)

}  // namespace copy_gather_scatter
