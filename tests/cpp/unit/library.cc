/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/library.h>

#include <legate.h>

#include <legate/redop/redop.h>

#include <gtest/gtest.h>

#include <string_view>
#include <utilities/utilities.h>

namespace test_library {

namespace {

class LibraryMapper : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& /*task*/,
    const std::vector<legate::mapping::StoreTarget>& /*options*/) override
  {
    return {};
  }

  std::optional<std::size_t> allocation_pool_size(
    const legate::mapping::Task& /*task*/, legate::mapping::StoreTarget /*memory_kind*/) override
  {
    return std::nullopt;
  }
};

}  // namespace

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
    {legate::VariantCode::CPU, legate::VariantOptions{}.with_has_allocations(true)},
    {legate::VariantCode::GPU,
     legate::VariantOptions{}.with_concurrent(true).with_elide_device_ctx_sync(true)}};
  const auto lib2 = runtime->create_library("test_library.bar", {}, nullptr, default_options2);

  // Creation of lib2 should not affect lib1
  ASSERT_EQ(lib1.get_default_variant_options(), default_options1);

  ASSERT_EQ(lib2.get_default_variant_options(), default_options2);
  // Repeated calls should get the same thing
  ASSERT_EQ(lib2.get_default_variant_options(), default_options2);
  // Creation of lib2 should not affect lib1
  ASSERT_EQ(lib1.get_default_variant_options(), default_options1);
}

TEST_F(Library, CreateNegative)
{
  constexpr std::string_view LIBNAME = "test_library.create_negative";

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME);

  ASSERT_EQ(lib.get_library_name(), LIBNAME);

  ASSERT_THAT([&] { static_cast<void>(runtime->create_library(LIBNAME)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr(fmt::format("Library {} already exists", LIBNAME))));
}

}  // namespace test_library

namespace example {

using Library = DefaultFixture;

namespace {

/// [Foo declaration]
class Foo : public legate::LegateTask<Foo> {
 public:
  // Foo declares a local task ID of 10
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{10}};

  static void cpu_variant(legate::TaskContext /* ctx */)
  {
    // some very useful work...
  }
};

}  // namespace

/// [Foo declaration]

TEST_F(Library, TaskIDExample)
{
  constexpr auto BAR_LIBNAME = std::string_view{"test_library.example.bar_lib"};
  const auto runtime         = legate::Runtime::get_runtime();

  // We don't care about const below, as it muddies up the example with pointless noise
  // NOLINTBEGIN(misc-const-correctness)

  /// [TaskID registration]
  legate::Library bar_lib = runtime->create_library(BAR_LIBNAME);

  // Foo registers itself with bar, claiming the bar-local task ID of 10.
  Foo::register_variants(bar_lib);

  // We can query information about the task (such as its name) using its local ID.
  auto task_name = bar_lib.get_task_name(Foo::TASK_CONFIG.task_id());

  ASSERT_EQ(task_name, "example::(anonymous namespace)::Foo");
  /// [TaskID registration]
  // NOLINTEND(misc-const-correctness)
}

}  // namespace example
