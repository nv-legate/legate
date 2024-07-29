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

namespace test_variant_options_precedence {

namespace {

constexpr std::size_t REG_RETURN_SIZE         = 12;
constexpr std::size_t DECL_RETURN_SIZE        = 34;
constexpr std::size_t LIB_DEFAULT_RETURN_SIZE = 56;

}  // namespace

struct HasDeclOptions : public legate::LegateTask<HasDeclOptions> {
  static constexpr auto TASK_ID = legate::LocalTaskID{1};
  static constexpr auto CPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_return_size(DECL_RETURN_SIZE);
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
    {{legate::VariantCode::CPU,
      legate::VariantOptions{}.with_return_size(LIB_DEFAULT_RETURN_SIZE)}});
}

void check_return_size(const legate::Library& library,
                       legate::LocalTaskID task_id,
                       std::size_t return_size_to_match)
{
  auto&& vinfo = library.find_task(task_id)->find_variant(legate::VariantCode::CPU);
  ASSERT_TRUE(vinfo.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(vinfo->get().options.return_size, return_size_to_match);
}

using VariantOptionsPrecedence = DefaultFixture;

TEST_F(VariantOptionsPrecedence, RegOptions)
{
  auto library = create_library("test_reg_options");
  HasDeclOptions::register_variants(
    library,
    {{legate::VariantCode::CPU, legate::VariantOptions{}.with_return_size(REG_RETURN_SIZE)}});
  check_return_size(library, HasDeclOptions::TASK_ID, REG_RETURN_SIZE);
}

TEST_F(VariantOptionsPrecedence, DeclOptions)
{
  auto library = create_library("test_decl_options");
  HasDeclOptions::register_variants(library);
  check_return_size(library, HasDeclOptions::TASK_ID, DECL_RETURN_SIZE);
}

TEST_F(VariantOptionsPrecedence, LibDefaultOptions)
{
  auto library = create_library("test_lib_default_options");
  NoDeclOptions::register_variants(library);
  check_return_size(library, NoDeclOptions::TASK_ID, LIB_DEFAULT_RETURN_SIZE);
}

TEST_F(VariantOptionsPrecedence, GlobalDefaultOptions)
{
  auto library = legate::Runtime::get_runtime()->create_library("test_global_default_options");
  NoDeclOptions::register_variants(library);
  check_return_size(library, NoDeclOptions::TASK_ID, legate::VariantOptions{}.return_size);
}

}  // namespace test_variant_options_precedence
