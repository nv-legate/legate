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

#include "task_simple.h"

namespace task {

namespace simple {

Legion::Logger logger(library_name);

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  HelloTask::register_variants(context);
  WriterTask::register_variants(context);
  ReducerTask::register_variants(context);
}

/*static*/ void HelloTask::cpu_variant(legate::TaskContext& context)
{
  auto& output = context.outputs()[0];
  auto shape   = output.shape<2>();

  if (shape.empty()) return;

  auto acc = output.write_accessor<int64_t, 2>(shape);
  for (legate::PointInRectIterator<2> it(shape); it.valid(); ++it)
    acc[*it] = (*it)[0] + (*it)[1] * 1000;
}

/*static*/ void WriterTask::cpu_variant(legate::TaskContext& context)
{
  auto& output1 = context.outputs()[0];
  auto& output2 = context.outputs()[1];

  auto acc1 = output1.write_accessor<int8_t, 2>();
  auto acc2 = output2.write_accessor<int32_t, 3>();

  acc1[{0, 0}]    = 10;
  acc2[{0, 0, 0}] = 20;
}

/*static*/ void ReducerTask::cpu_variant(legate::TaskContext& context)
{
  auto& red1 = context.reductions()[0];
  auto& red2 = context.reductions()[1];

  auto acc1 = red1.reduce_accessor<legate::SumReduction<int8_t>, true, 2>();
  auto acc2 = red2.reduce_accessor<legate::ProdReduction<int32_t>, true, 3>();

  acc1[{0, 0}].reduce(10);
  acc2[{0, 0, 0}].reduce(2);
}

}  // namespace simple

}  // namespace task
