/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <memory>
#include <utilities/utilities.h>

namespace test_variant_options_precedence {

namespace {

constexpr auto REG_OPTS =
  legate::VariantOptions{}.with_has_allocations(true).with_elide_device_ctx_sync(true);
constexpr auto DECL_OPTS        = legate::VariantOptions{}.with_elide_device_ctx_sync(true);
constexpr auto LIB_DEFAULT_OPTS = legate::VariantOptions{}.with_has_allocations(true);

struct HasDeclOptions : public legate::LegateTask<HasDeclOptions> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};
  static constexpr auto CPU_VARIANT_OPTIONS = DECL_OPTS;

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

struct HasDeclOptionsAndConfig : public legate::LegateTask<HasDeclOptionsAndConfig> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_variant_options(
      legate::VariantOptions{}.with_elide_device_ctx_sync(false));

  static constexpr auto CPU_VARIANT_OPTIONS = DECL_OPTS;

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

struct NoDeclOptions : public legate::LegateTask<NoDeclOptions> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

struct NoDeclOptionsAndConfig : public legate::LegateTask<NoDeclOptionsAndConfig> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_variant_options(DECL_OPTS);

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
  auto&& vinfo = library.find_task(task_id).find_variant(legate::VariantCode::CPU);
  ASSERT_TRUE(vinfo.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(vinfo->options(), options_to_match);
}

using VariantOptionsPrecedence = DefaultFixture;

}  // namespace

TEST_F(VariantOptionsPrecedence, RegOptions)
{
  auto library = create_library("test_reg_options");
  HasDeclOptions::register_variants(library, {{legate::VariantCode::CPU, REG_OPTS}});
  check_options(library, HasDeclOptions::TASK_CONFIG.task_id(), REG_OPTS);
}

TEST_F(VariantOptionsPrecedence, DeclOptions)
{
  auto library = create_library("test_decl_options");
  HasDeclOptions::register_variants(library);
  check_options(library, HasDeclOptions::TASK_CONFIG.task_id(), DECL_OPTS);
}

TEST_F(VariantOptionsPrecedence, DeclOptionsAndConfig)
{
  auto library = create_library("test_decl_options_and_config");
  HasDeclOptionsAndConfig::register_variants(library);
  check_options(library, HasDeclOptionsAndConfig::TASK_CONFIG.task_id(), DECL_OPTS);
}

TEST_F(VariantOptionsPrecedence, TaskConfigDefaultOptions)
{
  auto library = create_library("test_task_config_default_options");
  NoDeclOptionsAndConfig::register_variants(library);
  check_options(library, NoDeclOptionsAndConfig::TASK_CONFIG.task_id(), DECL_OPTS);
}

TEST_F(VariantOptionsPrecedence, LibDefaultOptions)
{
  auto library = create_library("test_lib_default_options");
  NoDeclOptions::register_variants(library);
  check_options(library, NoDeclOptions::TASK_CONFIG.task_id(), LIB_DEFAULT_OPTS);
}

TEST_F(VariantOptionsPrecedence, GlobalDefaultOptions)
{
  auto library = legate::Runtime::get_runtime()->create_library("test_global_default_options");
  NoDeclOptions::register_variants(library);
  check_options(
    library, NoDeclOptions::TASK_CONFIG.task_id(), legate::VariantOptions::DEFAULT_OPTIONS);
}

}  // namespace test_variant_options_precedence
