/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>

#include "copy_util.inl"
#include "legate.h"

namespace copy_scatter {

static const char* library_name = "test_copy_scatter";
static legate::Logger logger(library_name);

constexpr int32_t CHECK_SCATTER_TASK = FILL_INDIRECT_TASK + TEST_MAX_DIM * TEST_MAX_DIM;

template <int32_t IND_DIM, int32_t TGT_DIM>
struct CheckScatterTask : public legate::LegateTask<CheckScatterTask<IND_DIM, TGT_DIM>> {
  struct CheckScatterTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::TaskContext& context)
    {
      using VAL = legate::legate_type_of<CODE>;

      auto& src_store = context.inputs().at(0);
      auto& tgt_store = context.inputs().at(1);
      auto& ind_store = context.inputs().at(2);
      auto init       = context.scalars().at(0).value<VAL>();

      auto ind_shape = ind_store.shape<IND_DIM>();
      if (ind_shape.empty()) return;

      auto tgt_shape = tgt_store.shape<TGT_DIM>();

      legate::Buffer<bool, TGT_DIM> mask(tgt_shape, legate::Memory::Kind::SYSTEM_MEM);
      for (legate::PointInRectIterator<TGT_DIM> it(tgt_shape); it.valid(); ++it) mask[*it] = false;

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
        if (mask[p]) continue;
        EXPECT_EQ(tgt_acc[p], init);
      }
    }
  };

  static const int32_t TASK_ID = CHECK_SCATTER_TASK + IND_DIM * TEST_MAX_DIM + TGT_DIM;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto type_code = context.inputs().at(0).type().code;
    type_dispatch_for_test(type_code, CheckScatterTaskBody{}, context);
  }
};

void register_tasks()
{
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

  // CheckScatterTask<1, 1>::register_variants(context);
  CheckScatterTask<1, 2>::register_variants(context);
  // CheckScatterTask<1, 3>::register_variants(context);
  // CheckScatterTask<2, 1>::register_variants(context);
  CheckScatterTask<2, 2>::register_variants(context);
  CheckScatterTask<2, 3>::register_variants(context);
  CheckScatterTask<3, 1>::register_variants(context);
  CheckScatterTask<3, 2>::register_variants(context);
  // CheckScatterTask<3, 3>::register_variants(context);
}

struct ScatterSpec {
  std::vector<size_t> ind_shape;
  std::vector<size_t> data_shape;
  legate::Scalar seed;
  legate::Scalar init;

  std::string to_string() const
  {
    std::stringstream ss;
    auto print_shape = [&](auto& shape) {
      ss << "(";
      for (auto& ext : shape) ss << ext << ",";
      ss << ")";
    };
    ss << "indirection/source shape: " << ::to_string(ind_shape)
       << ", target shape: " << ::to_string(data_shape);
    ss << ", data type: " << seed.type().to_string();
    return std::move(ss).str();
  }
};

void check_scatter_output(legate::LibraryContext* context,
                          const legate::LogicalStore& src,
                          const legate::LogicalStore& tgt,
                          const legate::LogicalStore& ind,
                          const legate::Scalar& init)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();

  int32_t task_id = CHECK_SCATTER_TASK + ind.dim() * TEST_MAX_DIM + tgt.dim();

  auto task = runtime->create_task(context, task_id);

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
  auto context = runtime->find_library(library_name);

  auto& type = spec.seed.type();
  auto src   = runtime->create_store(spec.ind_shape, type.clone());
  auto tgt   = runtime->create_store(spec.data_shape, type.clone());
  auto ind   = runtime->create_store(spec.ind_shape, legate::point_type(spec.data_shape.size()));

  fill_input(context, src, spec.seed);
  fill_indirect(context, ind, tgt);
  runtime->issue_fill(tgt, spec.init);
  runtime->issue_scatter(tgt, ind, src);

  check_scatter_output(context, src, tgt, ind, spec.init);
}

// Note that the volume of indirection field should be smaller than that of the target to avoid
// duplicate updates on the same element, whose semantics is undefined.
TEST(Copy, Scatter1Dto2D)
{
  legate::Core::perform_registration<register_tasks>();
  std::vector<size_t> shape1d{5};
  test_scatter(
    ScatterSpec{shape1d, {7, 11}, legate::Scalar(int64_t(123)), legate::Scalar(int64_t(42))});
}

TEST(Copy, Scatter2Dto3D)
{
  legate::Core::perform_registration<register_tasks>();
  test_scatter(
    ScatterSpec{{3, 7}, {3, 6, 5}, legate::Scalar(uint32_t(456)), legate::Scalar(uint32_t(42))});
}

TEST(Copy, Scatter2Dto2D)
{
  legate::Core::perform_registration<register_tasks>();
  test_scatter(
    ScatterSpec{{4, 5}, {10, 11}, legate::Scalar(int64_t(12)), legate::Scalar(int64_t(42))});
}

TEST(Copy, Scatter3Dto2D)
{
  legate::Core::perform_registration<register_tasks>();
  test_scatter(
    ScatterSpec{{10, 10, 10}, {200, 200}, legate::Scalar(int64_t(1)), legate::Scalar(int64_t(42))});
}

}  // namespace copy_scatter
