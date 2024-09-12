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

#include "legate/type/type_info.h"

#include "copy_util.inl"
#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

// extern so that compilers don't also complain that function is unused!
extern void silence_unused_function_warnings()
{
  // defined in copy_util.inl
  static_cast<void>(::fill_indirect);
}

namespace copy_normal {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::int32_t TEST_MAX_DIM              = 3;
constexpr std::int32_t CHECK_COPY_TASK           = FILL_TASK + TEST_MAX_DIM;
constexpr std::int32_t CHECK_COPY_REDUCTION_TASK = CHECK_COPY_TASK + TEST_MAX_DIM;

}  // namespace

template <std::int32_t DIM>
struct CheckCopyTask : public legate::LegateTask<CheckCopyTask<DIM>> {
  struct CheckCopyTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::PhysicalStore& source,
                    legate::PhysicalStore& target,
                    legate::Rect<DIM>& shape)
    {
      using VAL = legate::type_of_t<CODE>;
      auto src  = source.read_accessor<VAL, DIM>(shape);
      auto tgt  = target.read_accessor<VAL, DIM>(shape);
      for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
        EXPECT_EQ(src[*it], tgt[*it]);
      }
    }
  };

  static constexpr auto TASK_ID = legate::LocalTaskID{CHECK_COPY_TASK + DIM};
  static void cpu_variant(legate::TaskContext context)
  {
    auto source = context.input(0).data();
    auto target = context.input(1).data();
    auto shape  = source.shape<DIM>();

    if (shape.empty()) {
      return;
    }

    type_dispatch(source.type().code(), CheckCopyTaskBody{}, source, target, shape);
  }
};

template <std::int32_t DIM>
struct CheckCopyReductionTask : public legate::LegateTask<CheckCopyReductionTask<DIM>> {
  struct CheckCopyReductionTaskBody {
    template <legate::Type::Code CODE, std::enable_if_t<legate::is_integral<CODE>::value, int> = 0>
    void operator()(legate::PhysicalStore& source,
                    legate::PhysicalStore& target,
                    const legate::Scalar& seed,
                    legate::Rect<DIM>& shape)
    {
      using VAL     = legate::type_of_t<CODE>;
      auto src      = source.read_accessor<VAL, DIM>(shape);
      auto tgt      = target.read_accessor<VAL, DIM>(shape);
      std::size_t i = 1;
      for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it, ++i) {
        EXPECT_EQ(src[*it] + i * seed.value<VAL>(), tgt[*it]);
      }
    }
    template <legate::Type::Code CODE, std::enable_if_t<!legate::is_integral<CODE>::value, int> = 0>
    void operator()(legate::PhysicalStore& /*source*/,
                    legate::PhysicalStore& /*target*/,
                    const legate::Scalar& /*seed*/,
                    legate::Rect<DIM>& /*shape*/)
    {
      LEGATE_ASSERT(false);
    }
  };

  static constexpr auto TASK_ID = legate::LocalTaskID{CHECK_COPY_REDUCTION_TASK + DIM};

  static void cpu_variant(legate::TaskContext context)
  {
    auto source = legate::PhysicalStore{context.input(0)};
    auto target = legate::PhysicalStore{context.input(1)};
    auto seed   = context.scalar(0);
    auto shape  = target.shape<DIM>();

    if (shape.empty()) {
      return;
    }

    type_dispatch(target.type().code(), CheckCopyReductionTaskBody{}, source, target, seed, shape);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_copy_normal";
  static void registration_callback(legate::Library library)
  {
    FillTask<1>::register_variants(library);
    FillTask<2>::register_variants(library);
    FillTask<3>::register_variants(library);
    CheckCopyTask<1>::register_variants(library);
    CheckCopyTask<2>::register_variants(library);
    CheckCopyTask<3>::register_variants(library);
    CheckCopyReductionTask<1>::register_variants(library);
    CheckCopyReductionTask<2>::register_variants(library);
    CheckCopyReductionTask<3>::register_variants(library);
  }
};

class NormalCopy : public RegisterOnceFixture<Config> {};

void check_copy_output(legate::Library library,
                       const legate::LogicalStore& src,
                       const legate::LogicalStore& tgt)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(
    library, legate::LocalTaskID{static_cast<std::int64_t>(CHECK_COPY_TASK) + tgt.dim()});

  auto src_part = task.declare_partition();
  auto tgt_part = task.declare_partition();

  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_constraint(legate::align(src_part, tgt_part));

  runtime->submit(std::move(task));
}

void check_copy_reduction_output(legate::Library library,
                                 const legate::LogicalStore& src,
                                 const legate::LogicalStore& tgt,
                                 const legate::Scalar& seed)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(
    library, legate::LocalTaskID{static_cast<std::int64_t>(CHECK_COPY_REDUCTION_TASK) + tgt.dim()});

  auto src_part = task.declare_partition();
  auto tgt_part = task.declare_partition();

  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_constraint(legate::align(src_part, tgt_part));
  task.add_scalar_arg(seed);

  runtime->submit(std::move(task));
}

struct NormalCopySpec {
  std::vector<std::uint64_t> shape;
  legate::Type type;
  legate::Scalar seed;
};

template <typename T>
struct NormalCopyReductionSpec {
  std::vector<std::uint64_t> shape;
  legate::Type type;
  legate::Scalar seed;
  T redop;
};

void test_normal_copy(const NormalCopySpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto& [shape, type, seed] = spec;

  auto input  = runtime->create_store(legate::Shape{shape}, type, true /*optimize_scalar*/);
  auto output = runtime->create_store(legate::Shape{shape}, type, true /*optimize_scalar*/);

  fill_input(library, input, seed);
  runtime->issue_copy(output, input);

  // check the result of copy
  check_copy_output(library, input, output);
}

template <typename T>
void test_normal_copy_reduction(const NormalCopyReductionSpec<T>& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto& [shape, type, seed, redop] = spec;

  auto input  = runtime->create_store(legate::Shape{shape}, type);
  auto output = runtime->create_store(legate::Shape{shape}, type);

  fill_input(library, input, seed);
  fill_input(library, output, seed);
  runtime->issue_copy(output, input, redop);

  // check the result of copy reduction
  check_copy_reduction_output(library, input, output, seed);
}

TEST_F(NormalCopy, Single)
{
  test_normal_copy({{4, 7}, legate::int64(), legate::Scalar{std::int64_t{12}}});
  test_normal_copy({{1000, 100}, legate::uint32(), legate::Scalar{std::uint32_t{3}}});
  test_normal_copy({{1}, legate::int64(), legate::Scalar{std::int64_t{12}}});
}

TEST_F(NormalCopy, SingleReduction)
{
  const std::vector<std::uint64_t> shape1{4, 7};
  const legate::Type type1{legate::int64()};
  const legate::Scalar seed1{std::int64_t{12}};
  const std::vector<std::uint64_t> shape2{1000, 100};
  const legate::Type type2{legate::uint32()};
  const legate::Scalar seed2{std::uint32_t{3}};
  const legate::ReductionOpKind redop{legate::ReductionOpKind::ADD};

  // Test with redop as ReductionOpKind
  test_normal_copy_reduction<legate::ReductionOpKind>(
    NormalCopyReductionSpec<legate::ReductionOpKind>{shape1, type1, seed1, redop});
  test_normal_copy_reduction<legate::ReductionOpKind>(
    NormalCopyReductionSpec<legate::ReductionOpKind>{shape2, type2, seed2, redop});
}

TEST_F(NormalCopy, SingleReductionInt32)
{
  const std::vector<std::uint64_t> shape1{4, 7};
  const legate::Type type1{legate::int64()};
  const legate::Scalar seed1{std::int64_t{12}};
  const std::vector<std::uint64_t> shape2{1000, 100};
  const legate::Type type2{legate::uint32()};
  const legate::Scalar seed2{std::uint32_t{3}};
  // ReductionOpKind::ADD
  const std::int32_t redop{0};

  static_assert(redop == static_cast<std::int32_t>(legate::ReductionOpKind::ADD));
  static_assert(std::is_same_v<std::int32_t, std::underlying_type_t<legate::ReductionOpKind>>);

  // Test with redop as int32
  test_normal_copy_reduction<std::int32_t>(
    NormalCopyReductionSpec<std::int32_t>{shape1, type1, seed1, redop});
  test_normal_copy_reduction<std::int32_t>(
    NormalCopyReductionSpec<std::int32_t>{shape2, type2, seed2, redop});
}

// NOLINTEND(readability-magic-numbers)

}  // namespace copy_normal
