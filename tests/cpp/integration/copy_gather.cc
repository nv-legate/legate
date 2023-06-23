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

namespace copy_gather {

static const char* library_name = "test_copy_gather";
static legate::Logger logger(library_name);

constexpr int32_t CHECK_GATHER_TASK = FILL_INDIRECT_TASK + TEST_MAX_DIM * TEST_MAX_DIM;

template <int32_t IND_DIM, int32_t SRC_DIM>
struct CheckGatherTask : public legate::LegateTask<CheckGatherTask<IND_DIM, SRC_DIM>> {
  struct CheckGatherTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::TaskContext& context)
    {
      using VAL = legate::legate_type_of<CODE>;

      auto& src_store = context.inputs().at(0);
      auto& tgt_store = context.inputs().at(1);
      auto& ind_store = context.inputs().at(2);

      auto ind_shape = ind_store.shape<IND_DIM>();
      if (ind_shape.empty()) return;

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

  static const int32_t TASK_ID = CHECK_GATHER_TASK + IND_DIM * TEST_MAX_DIM + SRC_DIM;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto type_code = context.inputs().at(0).type().code;
    type_dispatch_for_test(type_code, CheckGatherTaskBody{}, context);
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

void check_gather_output(legate::LibraryContext* context,
                         const legate::LogicalStore& src,
                         const legate::LogicalStore& tgt,
                         const legate::LogicalStore& ind)
{
  auto runtime    = legate::Runtime::get_runtime();
  auto machine    = runtime->get_machine();
  int32_t task_id = CHECK_GATHER_TASK + ind.dim() * TEST_MAX_DIM + src.dim();
  auto task       = runtime->create_task(context, task_id);
  auto src_part   = task->declare_partition();
  auto tgt_part   = task->declare_partition();
  auto ind_part   = task->declare_partition();
  task->add_input(src, src_part);
  task->add_input(tgt, tgt_part);
  task->add_input(ind, ind_part);

  task->add_constraint(legate::broadcast(src_part, legate::from_range<int32_t>(src.dim())));
  task->add_constraint(legate::broadcast(tgt_part, legate::from_range<int32_t>(tgt.dim())));
  task->add_constraint(legate::broadcast(ind_part, legate::from_range<int32_t>(ind.dim())));

  runtime->submit(std::move(task));
}

struct GatherSpec {
  std::vector<size_t> ind_shape;
  std::vector<size_t> data_shape;
  legate::Scalar seed;

  std::string to_string() const
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
  logger.print() << "Gather Copy: " << spec.to_string();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto& type = spec.seed.type();
  auto src   = runtime->create_store(spec.data_shape, type.clone());
  auto tgt   = runtime->create_store(spec.ind_shape, type.clone());
  auto ind   = runtime->create_store(spec.ind_shape, legate::point_type(spec.data_shape.size()));

  fill_input(context, src, spec.seed);
  fill_indirect(context, ind, src);

  auto copy = runtime->create_copy();
  copy->add_input(src);
  copy->add_output(tgt);
  copy->add_source_indirect(ind);
  runtime->submit(std::move(copy));

  check_gather_output(context, src, tgt, ind);
}

TEST(Copy, Gather2Dto1D)
{
  legate::Core::perform_registration<register_tasks>();
  std::vector<size_t> shape1d{5};
  test_gather(GatherSpec{shape1d, {7, 11}, legate::Scalar(int64_t(123))});
}

TEST(Copy, Gather3Dto2D)
{
  legate::Core::perform_registration<register_tasks>();
  test_gather(GatherSpec{{3, 7}, {3, 2, 5}, legate::Scalar(uint32_t(456))});
}

TEST(Copy, Gather1Dto3D)
{
  legate::Core::perform_registration<register_tasks>();
  std::vector<size_t> shape1d{5};
  test_gather(GatherSpec{{2, 5, 4}, shape1d, legate::Scalar(789.0)});
}

TEST(Copy, Gather2Dto2D)
{
  legate::Core::perform_registration<register_tasks>();
  test_gather(GatherSpec{{4, 5}, {10, 11}, legate::Scalar(int64_t(12))});
}

TEST(Copy, Gather2Dto3D)
{
  legate::Core::perform_registration<register_tasks>();
  test_gather(GatherSpec{{100, 100, 100}, {10, 10}, legate::Scalar(7.0)});
}

}  // namespace copy_gather
