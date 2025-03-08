/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <integration/copy_util.inl>
#include <utilities/utilities.h>

namespace copy_gather_scatter {

namespace {

constexpr std::int32_t CHECK_GATHER_SCATTER_TASK =
  FILL_INDIRECT_TASK + (TEST_MAX_DIM * TEST_MAX_DIM);

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
        ASSERT_EQ(copy, source);
        mask[tgt_point] = true;
      }

      for (legate::PointInRectIterator<TGT_DIM> it{tgt_shape}; it.valid(); ++it) {
        auto p = *it;
        if (mask[p]) {
          continue;
        }
        ASSERT_EQ(tgt_acc[p], init);
      }
    }
  };

  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CHECK_GATHER_SCATTER_TASK +
                                           (SRC_DIM * TEST_MAX_DIM * TEST_MAX_DIM) +
                                           (IND_DIM * TEST_MAX_DIM) + TGT_DIM}}
      .with_variant_options(legate::VariantOptions{}.with_has_allocations(true));

  static void cpu_variant(legate::TaskContext context)
  {
    auto type_code = context.input(0).type().code();
    type_dispatch_for_test(type_code, CheckGatherScatterTaskBody{}, context);
  }
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
                                                        (src.dim() * TEST_MAX_DIM * TEST_MAX_DIM) +
                                                        (src_ind.dim() * TEST_MAX_DIM) + tgt.dim());

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

void test_gather_scatter(const std::vector<std::uint64_t>& src_shape,
                         const std::vector<std::uint64_t>& ind_shape,
                         const std::vector<std::uint64_t>& tgt_shape,
                         const legate::Scalar& seed,
                         const legate::Scalar& init)
{
  LEGATE_ASSERT(seed.type() == init.type());

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto type = seed.type();
  auto src  = runtime->create_store(legate::Shape{src_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{tgt_shape}, type);
  auto src_ind =
    runtime->create_store(legate::Shape{ind_shape}, legate::point_type(src_shape.size()));
  auto tgt_ind =
    runtime->create_store(legate::Shape{ind_shape}, legate::point_type(tgt_shape.size()));

  fill_input(library, src, seed);
  fill_indirect(library, src_ind, src);
  fill_indirect(library, tgt_ind, tgt);
  runtime->issue_fill(tgt, init);

  runtime->issue_scatter_gather(tgt, tgt_ind, src, src_ind);

  check_gather_scatter_output(library, src, tgt, src_ind, tgt_ind, init);
}

template <typename T>
void test_gather_scatter_reduction(const std::vector<std::uint64_t>& src_shape,
                                   const std::vector<std::uint64_t>& ind_shape,
                                   const std::vector<std::uint64_t>& tgt_shape,
                                   const legate::Scalar& seed,
                                   const legate::Scalar& init,
                                   T redop)
{
  LEGATE_ASSERT(seed.type() == init.type());

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto type = seed.type();
  auto src  = runtime->create_store(legate::Shape{src_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{tgt_shape}, type);
  auto src_ind =
    runtime->create_store(legate::Shape{ind_shape}, legate::point_type(src_shape.size()));
  auto tgt_ind =
    runtime->create_store(legate::Shape{ind_shape}, legate::point_type(tgt_shape.size()));

  fill_input(library, src, seed);
  fill_indirect(library, src_ind, src);
  fill_indirect(library, tgt_ind, tgt);
  runtime->issue_fill(tgt, init);

  runtime->issue_scatter_gather(tgt, tgt_ind, src, src_ind, redop);

  check_gather_scatter_output(library, src, tgt, src_ind, tgt_ind, init);
}

}  // namespace

TEST_F(ScatterGatherCopy, 1Dto3Dvia2D)
{
  const std::vector<std::uint64_t> src_shape{5};
  const std::vector<std::uint64_t> ind_shape{7, 11};
  const std::vector<std::uint64_t> tgt_shape{10, 10, 10};
  const legate::Scalar seed{std::int64_t{123}};
  const legate::Scalar init{std::int64_t{42}};

  test_gather_scatter(src_shape, ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterGatherCopy, 2Dto1Dvia3D)
{
  const std::vector<std::uint64_t> src_shape{3, 7};
  const std::vector<std::uint64_t> ind_shape{3, 6, 5};
  const std::vector<std::uint64_t> tgt_shape{1000};
  const legate::Scalar seed{std::int64_t{456}};
  const legate::Scalar init{std::int64_t{42}};

  test_gather_scatter(src_shape, ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterGatherCopy, 3Dto2Dvia1D)
{
  const std::vector<std::uint64_t> src_shape{4, 5, 2};
  const std::vector<std::uint64_t> ind_shape{100};
  const std::vector<std::uint64_t> tgt_shape{50, 50};
  const legate::Scalar seed{std::int64_t{12}};
  const legate::Scalar init{std::int64_t{42}};

  test_gather_scatter(src_shape, ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterGatherCopy, 3Dto3Dvia3D)
{
  const std::vector<std::uint64_t> src_shape{10, 10, 10};
  const std::vector<std::uint64_t> ind_shape{5, 4, 2};
  const std::vector<std::uint64_t> tgt_shape{10, 10, 10};
  const legate::Scalar seed{std::int64_t{1}};
  const legate::Scalar init{std::int64_t{42}};

  test_gather_scatter(src_shape, ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterGatherCopy, 2Dto3Dvia2D)
{
  const std::vector<std::uint64_t> src_shape{27, 33};
  const std::vector<std::uint64_t> ind_shape{11, 7};
  const std::vector<std::uint64_t> tgt_shape{132, 121, 3};
  const legate::Scalar seed{std::int64_t{2}};
  const legate::Scalar init{std::int64_t{84}};

  test_gather_scatter(src_shape, ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterGatherCopy, ReductionEnum2Dto2Dvia2D)
{
  const std::vector<std::uint64_t> src_shape{10, 10};
  const std::vector<std::uint64_t> ind_shape{0, 0};
  const std::vector<std::uint64_t> tgt_shape{10, 10};
  const legate::Scalar seed{std::int64_t{12}};
  const legate::Scalar init{std::int64_t{42}};
  constexpr legate::ReductionOpKind redop{legate::ReductionOpKind::ADD};

  test_gather_scatter_reduction(src_shape, ind_shape, tgt_shape, seed, init, redop);
}

TEST_F(ScatterGatherCopy, ReductionInt322Dto2Dvia2D)
{
  const std::vector<std::uint64_t> src_shape{10, 10};
  const std::vector<std::uint64_t> ind_shape{0, 0};
  const std::vector<std::uint64_t> tgt_shape{10, 10};
  const legate::Scalar seed{std::int64_t{12}};
  const legate::Scalar init{std::int64_t{42}};
  constexpr std::int32_t redop{0};

  static_assert(redop == static_cast<std::int32_t>(legate::ReductionOpKind::ADD));
  static_assert(std::is_same_v<std::int32_t, std::underlying_type_t<legate::ReductionOpKind>>);

  test_gather_scatter_reduction(src_shape, ind_shape, tgt_shape, seed, init, redop);
}

}  // namespace copy_gather_scatter
