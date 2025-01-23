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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>
#include <memory>

namespace test_variant_options_precedence {

namespace {

constexpr auto REG_OPTS =
  legate::VariantOptions{}.with_has_allocations(true).with_elide_device_ctx_sync(true);
constexpr auto DECL_OPTS        = legate::VariantOptions{}.with_elide_device_ctx_sync(true);
constexpr auto LIB_DEFAULT_OPTS = legate::VariantOptions{}.with_has_allocations(true);

struct HasDeclOptions : public legate::LegateTask<HasDeclOptions> {
  static constexpr auto TASK_ID             = legate::LocalTaskID{1};
  static constexpr auto CPU_VARIANT_OPTIONS = DECL_OPTS;
  static void cpu_variant(legate::TaskContext /*context*/) {}
};

struct NoDeclOptions : public legate::LegateTask<NoDeclOptions> {
  static constexpr auto TASK_ID = legate::LocalTaskID{1};
  static void cpu_variant(legate::TaskContext /*context*/) {}
};

legate::Library create_library(std::string_view library_name)
{
  return legate::Runtime::get_runtime()->create_library(
    library_name,
    legate::ResourceConfig{},
    nullptr,
    {{legate::VariantCode::CPU, LIB_DEFAULT_OPTS}});
}

void check_options(const legate::Library& library,
                   legate::LocalTaskID task_id,
                   const legate::VariantOptions& options_to_match)
{
  auto&& vinfo = library.find_task(task_id)->find_variant(legate::VariantCode::CPU);
  ASSERT_TRUE(vinfo.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(vinfo->get().options, options_to_match);
}

using VariantOptionsPrecedence = DefaultFixture;

}  // namespace

TEST_F(VariantOptionsPrecedence, RegOptions)
{
  auto library = create_library("test_reg_options");
  HasDeclOptions::register_variants(library, {{legate::VariantCode::CPU, REG_OPTS}});
  check_options(library, HasDeclOptions::TASK_ID, REG_OPTS);
}

TEST_F(VariantOptionsPrecedence, DeclOptions)
{
  auto library = create_library("test_decl_options");
  HasDeclOptions::register_variants(library);
  check_options(library, HasDeclOptions::TASK_ID, DECL_OPTS);
}

TEST_F(VariantOptionsPrecedence, LibDefaultOptions)
{
  auto library = create_library("test_lib_default_options");
  NoDeclOptions::register_variants(library);
  check_options(library, NoDeclOptions::TASK_ID, LIB_DEFAULT_OPTS);
}

TEST_F(VariantOptionsPrecedence, GlobalDefaultOptions)
{
  auto library = legate::Runtime::get_runtime()->create_library("test_global_default_options");
  NoDeclOptions::register_variants(library);
  check_options(library, NoDeclOptions::TASK_ID, legate::VariantOptions{});
}

}  // namespace test_variant_options_precedence
