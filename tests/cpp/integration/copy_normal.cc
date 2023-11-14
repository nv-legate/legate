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

#include "core/type/type_info.h"

#include "copy_util.inl"
#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

// extern so that compilers don't also complain that function is unused!
extern void silence_unused_function_warnings()
{
  // defined in copy_util.inl
  static_cast<void>(::to_string);
  static_cast<void>(::fill_indirect);
}

namespace copy_normal {

using Copy = DefaultFixture;

static const char* library_name = "test_copy_normal";

static constexpr int32_t TEST_MAX_DIM = 3;

constexpr int32_t CHECK_COPY_TASK = FILL_TASK + TEST_MAX_DIM;

template <int32_t DIM>
struct CheckCopyTask : public legate::LegateTask<CheckCopyTask<DIM>> {
  struct CheckCopyTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::PhysicalStore& source,
                    legate::PhysicalStore& target,
                    legate::Rect<DIM>& shape)
    {
      using VAL = legate::type_of<CODE>;
      auto src  = source.read_accessor<VAL, DIM>(shape);
      auto tgt  = target.read_accessor<VAL, DIM>(shape);
      for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
        EXPECT_EQ(src[*it], tgt[*it]);
      }
    }
  };

  static const int32_t TASK_ID = CHECK_COPY_TASK + DIM;
  static void cpu_variant(legate::TaskContext context)
  {
    auto source = context.input(0).data();
    auto target = context.input(1).data();
    auto shape  = source.shape<DIM>();

    if (shape.empty()) return;

    type_dispatch(source.type().code(), CheckCopyTaskBody{}, source, target, shape);
  }
};

constexpr int32_t CHECK_COPY_REDUCTION_TASK = CHECK_COPY_TASK + TEST_MAX_DIM;

template <int32_t DIM>
struct CheckCopyReductionTask : public legate::LegateTask<CheckCopyReductionTask<DIM>> {
  struct CheckCopyReductionTaskBody {
    template <legate::Type::Code CODE, std::enable_if_t<legate::is_integral<CODE>::value, int> = 0>
    void operator()(legate::PhysicalStore& source,
                    legate::PhysicalStore& target,
                    const legate::Scalar& seed,
                    legate::Rect<DIM>& shape)
    {
      using VAL = legate::type_of<CODE>;
      auto src  = source.read_accessor<VAL, DIM>(shape);
      auto tgt  = target.read_accessor<VAL, DIM>(shape);
      legate::PointInRectIterator<DIM> it(shape);
      size_t i = 1;
      for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it, ++i) {
        EXPECT_EQ(src[*it] + i * seed.value<VAL>(), tgt[*it]);
      }
    }
    template <legate::Type::Code CODE, std::enable_if_t<!legate::is_integral<CODE>::value, int> = 0>
    void operator()(legate::PhysicalStore& source,
                    legate::PhysicalStore& target,
                    const legate::Scalar& seed,
                    legate::Rect<DIM>& shape)
    {
      assert(false);
    }
  };

  static const int32_t TASK_ID = CHECK_COPY_REDUCTION_TASK + DIM;
  static void cpu_variant(legate::TaskContext context)
  {
    auto source = legate::PhysicalStore{context.input(0)};
    auto target = legate::PhysicalStore{context.input(1)};
    auto& seed  = context.scalar(0);
    auto shape  = target.shape<DIM>();

    if (shape.empty()) return;

    type_dispatch(target.type().code(), CheckCopyReductionTaskBody{}, source, target, seed, shape);
  }
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
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

void check_copy_output(legate::Library library,
                       const legate::LogicalStore& src,
                       const legate::LogicalStore& tgt)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(library, CHECK_COPY_TASK + tgt.dim());

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
  auto task    = runtime->create_task(library, CHECK_COPY_REDUCTION_TASK + tgt.dim());

  auto src_part = task.declare_partition();
  auto tgt_part = task.declare_partition();

  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_constraint(legate::align(src_part, tgt_part));
  task.add_scalar_arg(seed);

  runtime->submit(std::move(task));
}

struct NormalCopySpec {
  std::vector<size_t> shape;
  legate::Type type;
  legate::Scalar seed;
};

struct NormalCopyReductionSpec {
  std::vector<size_t> shape;
  legate::Type type;
  legate::Scalar seed;
  legate::ReductionOpKind redop;
};

void test_normal_copy(const NormalCopySpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  auto& [shape, type, seed] = spec;

  auto input  = runtime->create_store(legate::Shape{shape}, type, true /*optimize_scalar*/);
  auto output = runtime->create_store(legate::Shape{shape}, type, true /*optimize_scalar*/);

  fill_input(library, input, seed);
  runtime->issue_copy(output, input);

  // check the result of copy
  check_copy_output(library, input, output);
}

void test_normal_copy_reduction(const NormalCopyReductionSpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);

  auto& [shape, type, seed, redop] = spec;

  auto input  = runtime->create_store(legate::Shape{shape}, type);
  auto output = runtime->create_store(legate::Shape{shape}, type);

  fill_input(library, input, seed);
  fill_input(library, output, seed);
  runtime->issue_copy(output, input, redop);

  // check the result of copy reduction
  check_copy_reduction_output(library, input, output, seed);
}

TEST_F(Copy, Single)
{
  register_tasks();
  test_normal_copy({{4, 7}, legate::int64(), legate::Scalar(int64_t(12))});
  test_normal_copy({{1000, 100}, legate::uint32(), legate::Scalar(uint32_t(3))});
  test_normal_copy({{1}, legate::int64(), legate::Scalar(int64_t(12))});
}

TEST_F(Copy, SingleReduction)
{
  register_tasks();
  test_normal_copy_reduction(
    {{4, 7}, legate::int64(), legate::Scalar(int64_t(12)), legate::ReductionOpKind::ADD});
  test_normal_copy_reduction(
    {{1000, 100}, legate::uint32(), legate::Scalar(uint32_t(3)), legate::ReductionOpKind::ADD});
}

}  // namespace copy_normal
