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

#include "core/runtime/detail/library.h"

#include "core/utilities/detail/strtoll.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace test_library {

class LibraryMapper : public legate::mapping::Mapper {
  void set_machine(const legate::mapping::MachineQueryInterface* /*machine*/) override {}

  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& /*task*/,
    const std::vector<legate::mapping::TaskTarget>& options) override
  {
    return options.front();
  }

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
  const char* LIBNAME = "test_library.libA";
  auto* runtime       = legate::Runtime::get_runtime();
  auto lib            = runtime->create_library(LIBNAME);

  EXPECT_EQ(lib, runtime->find_library(LIBNAME));

  const auto found_lib = runtime->maybe_find_library(LIBNAME);

  ASSERT_TRUE(found_lib.has_value());
  // We check the optional above ^^^
  EXPECT_EQ(lib, found_lib.value());  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_STREQ(lib.get_library_name().c_str(), LIBNAME);
}

TEST_F(Library, FindOrCreate)
{
  const char* LIBNAME = "test_library.libB";

  auto* runtime = legate::Runtime::get_runtime();

  legate::ResourceConfig config;
  config.max_tasks = 1;

  bool created = false;
  auto p_lib1  = runtime->find_or_create_library(LIBNAME, config, nullptr, &created);
  EXPECT_TRUE(created);

  config.max_tasks = 2;
  auto p_lib2      = runtime->find_or_create_library(LIBNAME, config, nullptr, &created);
  EXPECT_FALSE(created);
  EXPECT_EQ(p_lib1, p_lib2);
  EXPECT_TRUE(p_lib2.valid_task_id(p_lib2.get_task_id(0)));
  EXPECT_FALSE(p_lib2.valid_task_id(p_lib2.get_task_id(1)));
}

TEST_F(Library, FindNonExistent)
{
  const char* LIBNAME = "test_library.libC";

  auto* runtime = legate::Runtime::get_runtime();

  EXPECT_THROW((void)runtime->find_library(LIBNAME), std::out_of_range);

  EXPECT_EQ(runtime->maybe_find_library(LIBNAME), std::nullopt);
}

TEST_F(Library, InvalidReductionOPID)
{
  using SumReduction_Int32 = legate::SumReduction<std::int32_t>;

  static constexpr const char* LIBNAME = "test_library.libD";

  auto* runtime              = legate::Runtime::get_runtime();
  auto lib                   = runtime->create_library(LIBNAME);
  auto local_id              = 0;
  const auto value           = std::getenv("REALM_BACKTRACE");
  const bool realm_backtrace = value != nullptr && legate::detail::safe_strtoll(value) != 0;

  if (realm_backtrace) {
    EXPECT_DEATH((void)lib.register_reduction_operator<SumReduction_Int32>(local_id), "");
  } else {
    EXPECT_EXIT((void)lib.register_reduction_operator<SumReduction_Int32>(local_id),
                ::testing::KilledBySignal(SIGABRT),
                "");
  }
}

TEST_F(Library, RegisterReductionOP)
{
  using SumReduction_Int32 = legate::SumReduction<std::int32_t>;

  static constexpr const char* LIBNAME = "test_library.libE";
  legate::ResourceConfig config;
  config.max_reduction_ops = 1;

  auto* runtime = legate::Runtime::get_runtime();
  auto lib      = runtime->create_library(LIBNAME, config);
  auto local_id = 0;
  auto id       = lib.register_reduction_operator<SumReduction_Int32>(local_id);

  EXPECT_TRUE(lib.valid_reduction_op_id(static_cast<Legion::ReductionOpID>(id)));
  EXPECT_EQ(lib.get_local_reduction_op_id(static_cast<Legion::ReductionOpID>(id)), local_id);
}

TEST_F(Library, RegisterMapper)
{
  static constexpr const char* LIBNAME = "test_library.libF";

  auto* runtime    = legate::Runtime::get_runtime();
  auto lib         = runtime->create_library(LIBNAME);
  auto* mapper_old = lib.impl()->get_legion_mapper();
  auto mapper      = std::make_unique<LibraryMapper>();
  lib.register_mapper(std::move(mapper));
  auto* mapper_new = lib.impl()->get_legion_mapper();
  EXPECT_NE(mapper_old, mapper_new);
}

}  // namespace test_library
