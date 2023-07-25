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

namespace test_registration {

template <int32_t ID>
struct CPUVariantTask : public legate::LegateTask<CPUVariantTask<ID>> {
  static const int32_t TASK_ID = ID;
  static void cpu_variant(legate::TaskContext& context) {}
};

template <int32_t ID>
struct GPUVariantTask : public legate::LegateTask<GPUVariantTask<ID>> {
  static const int32_t TASK_ID = ID;
  static void gpu_variant(legate::TaskContext& context) {}
};

}  // namespace test_registration

void test_duplicates()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto library  = runtime->create_library("libA");
  test_registration::CPUVariantTask<0>::register_variants(library);
  EXPECT_THROW(test_registration::CPUVariantTask<0>::register_variants(library),
               std::invalid_argument);
}

void test_out_of_bounds_task_id()
{
  legate::ResourceConfig config;
  config.max_tasks = 1;
  auto* runtime    = legate::Runtime::get_runtime();
  auto library     = runtime->create_library("libA", config);

  EXPECT_THROW(test_registration::CPUVariantTask<1>::register_variants(library), std::out_of_range);
}

TEST(Registration, Duplicate) { legate::Core::perform_registration<test_duplicates>(); }

TEST(Registration, TaskIDOutOfBounds)
{
  legate::Core::perform_registration<test_out_of_bounds_task_id>();
}
