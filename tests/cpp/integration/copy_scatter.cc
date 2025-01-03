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

namespace copy_scatter {

namespace {

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_copy_scatter";
  static void registration_callback(legate::Library library);
};

constexpr std::int32_t CHECK_SCATTER_TASK = FILL_INDIRECT_TASK + (TEST_MAX_DIM * TEST_MAX_DIM);

template <std::int32_t IND_DIM, std::int32_t TGT_DIM>
struct CheckScatterTask : public legate::LegateTask<CheckScatterTask<IND_DIM, TGT_DIM>> {
  static constexpr auto TASK_ID =
    legate::LocalTaskID{CHECK_SCATTER_TASK + (IND_DIM * TEST_MAX_DIM) + TGT_DIM};
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  struct CheckScatterTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::TaskContext context)
    {
      using VAL = legate::type_of_t<CODE>;

      auto src_store = context.input(0).data();
      auto tgt_store = context.input(1).data();
      auto ind_store = context.input(2).data();
      auto init      = context.scalar(0).value<VAL>();

      auto ind_shape = ind_store.shape<IND_DIM>();
      if (ind_shape.empty()) {
        return;
      }

      auto tgt_shape = tgt_store.shape<TGT_DIM>();

      const legate::Buffer<bool, TGT_DIM> mask{tgt_shape, legate::Memory::Kind::SYSTEM_MEM};
      for (legate::PointInRectIterator<TGT_DIM> it{tgt_shape}; it.valid(); ++it) {
        mask[*it] = false;
      }

      auto src_acc = src_store.read_accessor<VAL, IND_DIM>();
      auto tgt_acc = tgt_store.read_accessor<VAL, TGT_DIM>();
      auto ind_acc = ind_store.read_accessor<legate::Point<TGT_DIM>, IND_DIM>();

      for (legate::PointInRectIterator<IND_DIM> it{ind_shape}; it.valid(); ++it) {
        auto p      = ind_acc[*it];
        auto copy   = tgt_acc[p];
        auto source = src_acc[*it];
        ASSERT_EQ(copy, source);
        mask[p] = true;
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

  static void cpu_variant(legate::TaskContext context)
  {
    auto type_code = context.input(0).type().code();
    type_dispatch_for_test(type_code, CheckScatterTaskBody{}, context);
  }
};

/*static*/ void Config::registration_callback(legate::Library library)
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

class ScatterCopy : public RegisterOnceFixture<Config> {};

void check_scatter_output(legate::Library library,
                          const legate::LogicalStore& src,
                          const legate::LogicalStore& tgt,
                          const legate::LogicalStore& ind,
                          const legate::Scalar& init)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();

  const auto task_id =
    static_cast<legate::LocalTaskID>(CHECK_SCATTER_TASK + (ind.dim() * TEST_MAX_DIM) + tgt.dim());

  auto task = runtime->create_task(library, task_id);

  auto src_part = task.declare_partition();
  auto tgt_part = task.declare_partition();
  auto ind_part = task.declare_partition();
  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_input(ind, ind_part);
  task.add_scalar_arg(init);

  task.add_constraint(legate::broadcast(src_part, legate::from_range(src.dim())));
  task.add_constraint(legate::broadcast(tgt_part, legate::from_range(tgt.dim())));
  task.add_constraint(legate::broadcast(ind_part, legate::from_range(ind.dim())));

  runtime->submit(std::move(task));
}

void test_scatter(const std::vector<std::uint64_t>& ind_shape,
                  const std::vector<std::uint64_t>& tgt_shape,
                  const legate::Scalar& seed,
                  const legate::Scalar& init)
{
  LEGATE_ASSERT(seed.type() == init.type());

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto type = seed.type();
  auto src  = runtime->create_store(legate::Shape{ind_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{tgt_shape}, type);
  auto ind  = runtime->create_store(legate::Shape{ind_shape}, legate::point_type(tgt_shape.size()));

  fill_input(library, src, seed);
  fill_indirect(library, ind, tgt);
  runtime->issue_fill(tgt, init);
  runtime->issue_scatter(tgt, ind, src);

  check_scatter_output(library, src, tgt, ind, init);
}

template <typename T>
void test_scatter_reduction(const std::vector<std::uint64_t>& ind_shape,
                            const std::vector<std::uint64_t>& tgt_shape,
                            const legate::Scalar& seed,
                            const legate::Scalar& init,
                            T redop)
{
  LEGATE_ASSERT(seed.type() == init.type());

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto type = seed.type();
  auto src  = runtime->create_store(legate::Shape{ind_shape}, type);
  auto tgt  = runtime->create_store(legate::Shape{tgt_shape}, type);
  auto ind  = runtime->create_store(legate::Shape{ind_shape}, legate::point_type(tgt_shape.size()));

  fill_input(library, src, seed);
  fill_indirect(library, ind, tgt);
  runtime->issue_fill(tgt, init);
  runtime->issue_scatter(tgt, ind, src, redop);

  check_scatter_output(library, src, tgt, ind, init);
}

}  // namespace

// Note that the volume of indirection field should be smaller than that of the target to avoid
// duplicate updates on the same element, whose semantics is undefined.
TEST_F(ScatterCopy, 1Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{5};
  const std::vector<std::uint64_t> tgt_shape{7, 11};
  const legate::Scalar seed{std::int64_t{123}};
  const legate::Scalar init{std::int64_t{42}};

  test_scatter(ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterCopy, 2Dto3D)
{
  const std::vector<std::uint64_t> ind_shape{3, 7};
  const std::vector<std::uint64_t> tgt_shape{3, 6, 5};
  const legate::Scalar seed{std::uint32_t{456}};
  const legate::Scalar init{std::uint32_t{42}};

  test_scatter(ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterCopy, 2Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{4, 5};
  const std::vector<std::uint64_t> tgt_shape{10, 11};
  const legate::Scalar seed{std::int64_t{12}};
  const legate::Scalar init{std::int64_t{42}};

  test_scatter(ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterCopy, 3Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{10, 10, 10};
  const std::vector<std::uint64_t> tgt_shape{200, 200};
  const legate::Scalar seed{std::int64_t{1}};
  const legate::Scalar init{std::int64_t{42}};

  test_scatter(ind_shape, tgt_shape, seed, init);
}

TEST_F(ScatterCopy, ReductionEnum2Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{0, 0};
  const std::vector<std::uint64_t> tgt_shape{10, 11};
  const legate::Scalar seed{std::int64_t{12}};
  const legate::Scalar init{std::int64_t{42}};
  constexpr legate::ReductionOpKind redop{legate::ReductionOpKind::MAX};

  test_scatter_reduction(ind_shape, tgt_shape, seed, init, redop);
}

TEST_F(ScatterCopy, ReductionInt322Dto2D)
{
  const std::vector<std::uint64_t> ind_shape{0, 0};
  const std::vector<std::uint64_t> tgt_shape{10, 11};
  const legate::Scalar seed{std::int64_t{12}};
  const legate::Scalar init{std::int64_t{42}};
  constexpr std::int32_t redop{4};

  static_assert(redop == static_cast<std::int32_t>(legate::ReductionOpKind::MAX));
  static_assert(std::is_same_v<std::int32_t, std::underlying_type_t<legate::ReductionOpKind>>);

  test_scatter_reduction(ind_shape, tgt_shape, seed, init, redop);
}

}  // namespace copy_scatter
