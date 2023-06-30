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

#include "legate.h"

#include "copy_util.inl"

namespace copy_normal {

static const char* library_name = "test_copy_normal";

static constexpr int32_t TEST_MAX_DIM = 3;

constexpr int32_t CHECK_TASK = FILL_TASK + TEST_MAX_DIM;

template <int32_t DIM>
struct CheckTask : public legate::LegateTask<CheckTask<DIM>> {
  struct CheckTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::Store& source, legate::Store& target, legate::Rect<DIM>& shape)
    {
      using VAL = legate::legate_type_of<CODE>;
      auto src  = source.read_accessor<VAL, DIM>(shape);
      auto tgt  = target.read_accessor<VAL, DIM>(shape);
      for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) {
        EXPECT_EQ(src[*it], tgt[*it]);
      }
    }
  };

  static const int32_t TASK_ID = CHECK_TASK + DIM;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& source = context.inputs().at(0);
    auto& target = context.inputs().at(1);
    auto shape   = source.shape<DIM>();

    if (shape.empty()) return;

    type_dispatch(source.type().code, CheckTaskBody{}, source, target, shape);
  }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  FillTask<1>::register_variants(context);
  FillTask<2>::register_variants(context);
  FillTask<3>::register_variants(context);
  CheckTask<1>::register_variants(context);
  CheckTask<2>::register_variants(context);
  CheckTask<3>::register_variants(context);
}

void check_output(legate::LibraryContext* context,
                  const legate::LogicalStore& src,
                  const legate::LogicalStore& tgt)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(context, CHECK_TASK + tgt.dim());

  auto src_part = task.declare_partition();
  auto tgt_part = task.declare_partition();

  task.add_input(src, src_part);
  task.add_input(tgt, tgt_part);
  task.add_constraint(legate::align(src_part, tgt_part));

  runtime->submit(std::move(task));
}

const auto uint32  = legate::uint32();
const auto int64   = legate::int64();
const auto float64 = legate::float64();

struct NormalCopySpec {
  std::vector<size_t> shape;
  const legate::Type& type;
  legate::Scalar seed;
};

void test_normal_copy(const NormalCopySpec& spec)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto& [shape, type, seed] = spec;

  auto input  = runtime->create_store(shape, type.clone());
  auto output = runtime->create_store(shape, type.clone());

  fill_input(context, input, seed);
  runtime->issue_copy(output, input);

  // check the result of copy
  check_output(context, input, output);
}

TEST(Copy, Single)
{
  legate::Core::perform_registration<register_tasks>();
  // For some reason, clang-format gets tripped over by singleton initialization lists,
  // so factor out the definition here
  std::vector<size_t> shape1d{9};
  test_normal_copy({{4, 7}, *int64, legate::Scalar(int64_t(12))});
  test_normal_copy({{1000, 100}, *uint32, legate::Scalar(uint32_t(3))});
}

}  // namespace copy_normal
