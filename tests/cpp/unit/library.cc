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

#include "legate/runtime/detail/library.h"

#include "legate/utilities/detail/strtoll.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>
#include <string_view>

namespace test_library {

class LibraryMapper : public legate::mapping::Mapper {
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& /*task*/,
    const std::vector<legate::mapping::StoreTarget>& /*options*/) override
  {
    return {};
  }

  legate::Scalar tunable_value(legate::TunableID /*tunable_id*/) override
  {
    LEGATE_ABORT("This method should never be called");
    return legate::Scalar{};
  }
};

using Library = DefaultFixture;

TEST_F(Library, Create)
{
  constexpr std::string_view LIBNAME = "test_library.libA";

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME);

  ASSERT_EQ(lib, runtime->find_library(LIBNAME));

  const auto found_lib = runtime->maybe_find_library(LIBNAME);

  ASSERT_TRUE(found_lib.has_value());
  // We check the optional above ^^^
  ASSERT_EQ(lib, found_lib.value());  // NOLINT(bugprone-unchecked-optional-access)
  ASSERT_EQ(lib.get_library_name(), LIBNAME);
}

TEST_F(Library, FindOrCreate)
{
  constexpr std::string_view LIBNAME  = "test_library.libB";
  constexpr std::string_view LIBNAME1 = "test_library.libB_1";

  auto* runtime = legate::Runtime::get_runtime();

  legate::ResourceConfig config;
  config.max_tasks = 1;

  bool created = false;
  auto p_lib1  = runtime->find_or_create_library(LIBNAME, config, nullptr, {}, &created);
  ASSERT_TRUE(created);

  config.max_tasks = 2;
  auto p_lib2      = runtime->find_or_create_library(LIBNAME, config, nullptr, {}, &created);
  ASSERT_FALSE(created);
  ASSERT_EQ(p_lib1, p_lib2);
  ASSERT_TRUE(p_lib2.valid_task_id(p_lib2.get_task_id(legate::LocalTaskID{0})));
  ASSERT_THROW(static_cast<void>(p_lib2.get_task_id(legate::LocalTaskID{1})), std::out_of_range);

  auto p_lib3 = runtime->find_or_create_library(LIBNAME1, config, nullptr, {}, &created);
  ASSERT_TRUE(created);
  ASSERT_NE(p_lib1, p_lib3);
}

TEST_F(Library, FindNonExistent)
{
  constexpr std::string_view LIBNAME = "test_library.libC";

  auto* runtime = legate::Runtime::get_runtime();

  ASSERT_THROW(static_cast<void>(runtime->find_library(LIBNAME)), std::out_of_range);

  ASSERT_EQ(runtime->maybe_find_library(LIBNAME), std::nullopt);
}

TEST_F(Library, InvalidReductionOPID)
{
  using SumReduction_Int32 = legate::SumReduction<std::int32_t>;

  constexpr std::string_view LIBNAME = "test_library.libD";

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME);
  auto local_id = legate::LocalRedopID{0};
  ASSERT_THROW(static_cast<void>(lib.register_reduction_operator<SumReduction_Int32>(local_id)),
               std::out_of_range);
}

TEST_F(Library, RegisterReductionOP)
{
  using SumReduction_Int32 = legate::SumReduction<std::int32_t>;

  constexpr std::string_view LIBNAME = "test_library.libE";
  legate::ResourceConfig config;
  config.max_reduction_ops = 1;

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME, config);
  auto local_id = legate::LocalRedopID{0};
  auto id       = lib.register_reduction_operator<SumReduction_Int32>(local_id);

  ASSERT_TRUE(lib.valid_reduction_op_id(id));
  ASSERT_EQ(lib.get_local_reduction_op_id(id), local_id);
}

TEST_F(Library, TaskID)
{
  constexpr std::string_view LIBNAME = "test_library.libG";

  legate::ResourceConfig config;
  config.max_tasks = 1;

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME, config);

  auto local_task_id = legate::LocalTaskID{0};
  auto task_id       = lib.get_task_id(local_task_id);
  ASSERT_TRUE(lib.valid_task_id(task_id));
  ASSERT_EQ(lib.get_local_task_id(task_id), local_task_id);

  auto task_id_negative =
    legate::GlobalTaskID{static_cast<std::underlying_type_t<decltype(task_id)>>(task_id) + 1};
  ASSERT_FALSE(lib.valid_task_id(task_id_negative));
}

TEST_F(Library, ProjectID)
{
  constexpr std::string_view LIBNAME = "test_library.libH";

  legate::ResourceConfig config;
  config.max_projections = 2;

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME, config);

  auto local_proj_id_1 = 0;
  auto proj_id_1       = lib.get_projection_id(local_proj_id_1);
  ASSERT_EQ(proj_id_1, 0);
  ASSERT_FALSE(lib.valid_projection_id(proj_id_1));

  auto local_proj_id_2 = 1;
  auto proj_id_2       = lib.get_projection_id(local_proj_id_2);
  ASSERT_TRUE(lib.valid_projection_id(proj_id_2));
  ASSERT_EQ(lib.get_local_projection_id(proj_id_2), local_proj_id_2);

  auto proj_id_negative = proj_id_1 + 2;
  ASSERT_FALSE(lib.valid_projection_id(proj_id_negative));
}

TEST_F(Library, ShardingID)
{
  constexpr std::string_view LIBNAME = "test_library.libI";

  legate::ResourceConfig config;
  config.max_shardings = 1;

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME, config);

  auto local_sharding_id = 0;
  auto sharding_id       = lib.get_sharding_id(local_sharding_id);
  ASSERT_TRUE(lib.valid_sharding_id(sharding_id));
  ASSERT_EQ(lib.get_local_sharding_id(sharding_id), local_sharding_id);

  auto sharding_id_negative = sharding_id + 1;
  ASSERT_FALSE(lib.valid_sharding_id(sharding_id_negative));
}

TEST_F(Library, ResourceIdScopeNegative)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  constexpr std::string_view LIBNAME = "test_library.libJ";

  // Test exception thrown in ResourceIdScope creation function
  legate::ResourceConfig config;
  config.max_tasks     = 1;
  config.max_dyn_tasks = 2;

  auto* runtime = legate::Runtime::get_runtime();
  ASSERT_THROW(static_cast<void>(runtime->create_library(LIBNAME, config)), std::out_of_range);
}

TEST_F(Library, GenerateIDNegative)
{
  constexpr std::string_view LIBNAME = "test_library.libK";

  // Test exception thrown in generate_id
  legate::ResourceConfig config;
  config.max_tasks     = 1;
  config.max_dyn_tasks = 0;

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME, config);
  ASSERT_THROW(static_cast<void>(lib.impl()->get_new_task_id()), std::overflow_error);
}

TEST_F(Library, VariantOptions)
{
  const auto runtime = legate::Runtime::get_runtime();

  const std::map<legate::VariantCode, legate::VariantOptions> default_options1 = {};
  const auto lib1 = runtime->create_library("test_library.foo", {}, nullptr, default_options1);

  ASSERT_EQ(lib1.get_default_variant_options(), default_options1);
  // Repeated calls should get the same thing
  ASSERT_EQ(lib1.get_default_variant_options(), default_options1);

  const std::map<legate::VariantCode, legate::VariantOptions> default_options2 = {
    {legate::VariantCode::CPU, legate::VariantOptions{}.with_return_size(1234)},
    {legate::VariantCode::GPU,
     legate::VariantOptions{}.with_idempotent(true).with_concurrent(true).with_return_size(
       7355608)}};
  const auto lib2 = runtime->create_library("test_library.bar", {}, nullptr, default_options2);

  // Creation of lib2 should not affect lib1
  ASSERT_EQ(lib1.get_default_variant_options(), default_options1);

  ASSERT_EQ(lib2.get_default_variant_options(), default_options2);
  // Repeated calls should get the same thing
  ASSERT_EQ(lib2.get_default_variant_options(), default_options2);
  // Creation of lib2 should not affect lib1
  ASSERT_EQ(lib1.get_default_variant_options(), default_options1);
}

}  // namespace test_library

namespace example {

using Library = DefaultFixture;

/// [Foo declaration]
class Foo : public legate::LegateTask<Foo> {
 public:
  // Foo declares a local task ID of 10
  static constexpr auto TASK_ID = legate::LocalTaskID{10};

  static void cpu_variant(legate::TaskContext /* ctx */)
  {
    // some very useful work...
  }
};
/// [Foo declaration]

TEST_F(Library, TaskIDExample)
{
  constexpr auto BAR_LIBNAME = std::string_view{"test_library.example.bar_lib"};
  constexpr auto BAZ_LIBNAME = std::string_view{"test_library.example.baz_lib"};
  const auto runtime         = legate::Runtime::get_runtime();

  // We don't care about const below, as it muddies up the example with pointless noise
  // NOLINTBEGIN(misc-const-correctness)

  /// [TaskID registration]
  legate::Library bar_lib = runtime->create_library(BAR_LIBNAME);
  legate::Library baz_lib = runtime->create_library(BAZ_LIBNAME);

  // Foo registers itself with bar, claiming the bar-local task ID of 10.
  Foo::register_variants(bar_lib);
  // Retrieve the global task ID after registration.
  legate::GlobalTaskID gid_bar = bar_lib.get_task_id(Foo::TASK_ID);

  // This should be false, Foo has not registered itself to baz yet.
  ASSERT_FALSE(baz_lib.valid_task_id(gid_bar));

  // However, we can query information from Legion about this task (such as its name), since
  // the global task ID has been assigned.
  const char* legion_task_name{};

  Legion::Runtime::get_runtime()->retrieve_name(static_cast<Legion::TaskID>(gid_bar),
                                                legion_task_name);
  ASSERT_STREQ(legion_task_name, "example::Foo");

  // We can get the same information using the local ID from the Library
  auto task_name = bar_lib.get_task_name(Foo::TASK_ID);

  ASSERT_EQ(task_name, legion_task_name);
  /// [TaskID registration]
  // NOLINTEND(misc-const-correctness)
}

}  // namespace example
