/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/cpus.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>

#include <realm/module_config.h>

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utilities/utilities.h>

namespace test_configure_cpus {

class ConfigureCPUsUnit : public DefaultFixture, public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(,
                         ConfigureCPUsUnit,
                         ::testing::Bool(),
                         ::testing::PrintToStringParamName{});

using OpenMPsType = legate::detail::Argument<std::int32_t>;
using UtilType    = OpenMPsType;
using GPUsType    = OpenMPsType;
using CPUsType    = OpenMPsType;

namespace {

class MockCoreModuleConfig : public Realm::ModuleConfig {
 public:
  MockCoreModuleConfig(std::int32_t num_cpus, bool should_fail)
    : ModuleConfig{"mock_core"}, num_cpus_{num_cpus}
  {
    // this will make it return error REALM_MODULE_CONFIG_ERROR_NO_RESOURCE when
    // get_resource is called
    resource_discover_finished = false;
    if (!should_fail) {
      const auto inserted = resource_map.insert({"cpu", &num_cpus_}).second;

      LEGATE_CHECK(inserted);
      resource_discover_finished = true;
    }
  }

 private:
  std::int32_t num_cpus_{};
};

}  // namespace

TEST_P(ConfigureCPUsUnit, Preset)
{
  constexpr auto NUM_CPUS = 10;
  const auto core         = MockCoreModuleConfig{8, /* should_fail */ false};
  const auto omps         = OpenMPsType{nullptr, "--omps", 1};
  const auto util         = UtilType{nullptr, "--util", 2};
  const auto gpus         = GPUsType{nullptr, "--gpus", 3};
  auto cpus               = CPUsType{nullptr, "--cpus", NUM_CPUS};

  legate::detail::configure_cpus(
    /* auto_config */ GetParam(), core, omps, util, gpus, &cpus);
  ASSERT_EQ(cpus.value(), NUM_CPUS);
}

TEST_F(ConfigureCPUsUnit, NoAutoConfig)
{
  const auto core = MockCoreModuleConfig{8, /* should_fail */ false};
  const auto omps = OpenMPsType{nullptr, "--omps", 1};
  const auto util = UtilType{nullptr, "--util", 2};
  const auto gpus = GPUsType{nullptr, "--gpus", 3};
  auto cpus       = CPUsType{nullptr, "--cpus", -1};

  legate::detail::configure_cpus(
    /* auto_config */ false, core, omps, util, gpus, &cpus);
  ASSERT_EQ(cpus.value(), 1);
}

TEST_F(ConfigureCPUsUnit, HaveOpenMP)
{
  const auto core = MockCoreModuleConfig{8, /* should_fail */ false};
  const auto omps = OpenMPsType{nullptr, "--omps", 1};
  const auto util = UtilType{nullptr, "--util", 2};
  const auto gpus = GPUsType{nullptr, "--gpus", 3};
  auto cpus       = CPUsType{nullptr, "--cpus", -1};

  legate::detail::configure_cpus(
    /* auto_config */ true, core, omps, util, gpus, &cpus);
  ASSERT_EQ(cpus.value(), 1);
}

TEST_F(ConfigureCPUsUnit, HaveGPUs)
{
  const auto core = MockCoreModuleConfig{8, /* should_fail */ false};
  const auto omps = OpenMPsType{nullptr, "--omps", 0};
  const auto util = UtilType{nullptr, "--util", 2};
  const auto gpus = GPUsType{nullptr, "--gpus", 3};
  auto cpus       = CPUsType{nullptr, "--cpus", -1};

  legate::detail::configure_cpus(
    /* auto_config */ true, core, omps, util, gpus, &cpus);
  ASSERT_EQ(cpus.value(), gpus.value());
}

TEST_F(ConfigureCPUsUnit, AutoConfig)
{
  constexpr auto NUM_CPUS = 8;
  const auto core         = MockCoreModuleConfig{NUM_CPUS, /* should_fail */ false};
  const auto omps         = OpenMPsType{nullptr, "--omps", 0};
  const auto util         = UtilType{nullptr, "--util", 2};
  const auto gpus         = GPUsType{nullptr, "--gpus", 0};
  auto cpus               = CPUsType{nullptr, "--cpus", -1};

  legate::detail::configure_cpus(
    /* auto_config */ true, core, omps, util, gpus, &cpus);

  const auto expected_cpus = NUM_CPUS - util.value() - gpus.value();

  ASSERT_EQ(cpus.value(), expected_cpus);
}

TEST_F(ConfigureCPUsUnit, AutoConfigFail)
{
  const auto core = MockCoreModuleConfig{8, /* should_fail */ true};
  const auto omps = OpenMPsType{nullptr, "--omps", 0};
  const auto util = UtilType{nullptr, "--util", 2};
  const auto gpus = GPUsType{nullptr, "--gpus", 0};
  auto cpus       = CPUsType{nullptr, "--cpus", -1};

  ASSERT_THAT(
    [&] {
      legate::detail::configure_cpus(
        /* auto_config */ true, core, omps, util, gpus, &cpus);
    },
    ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(
      ::testing::HasSubstr("Core Realm module could not determine the number of CPU cores.")));
  ASSERT_EQ(cpus.value(), -1);
}

TEST_F(ConfigureCPUsUnit, AutoConfigFailNoCPUs)
{
  const auto core = MockCoreModuleConfig{0, /* should_fail */ false};
  const auto omps = OpenMPsType{nullptr, "--omps", 0};
  const auto util = UtilType{nullptr, "--util", 2};
  const auto gpus = GPUsType{nullptr, "--gpus", 0};
  auto cpus       = CPUsType{nullptr, "--cpus", -1};

  ASSERT_THAT(
    [&] {
      legate::detail::configure_cpus(
        /* auto_config */ true, core, omps, util, gpus, &cpus);
    },
    ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(
      ::testing::HasSubstr("Core Realm module detected 0 CPU cores while configuring CPUs.")));
  ASSERT_EQ(cpus.value(), -1);
}

TEST_F(ConfigureCPUsUnit, AutoConfigFailNotEnoughCores)
{
  constexpr auto NUM_CPUS = 1;
  const auto core         = MockCoreModuleConfig{NUM_CPUS, /* should_fail */ false};
  const auto omps         = OpenMPsType{nullptr, "--omps", 0};
  const auto util         = UtilType{nullptr, "--util", 2};
  const auto gpus         = GPUsType{nullptr, "--gpus", 0};
  auto cpus               = CPUsType{nullptr, "--cpus", -1};

  ASSERT_THAT(
    [&] {
      legate::detail::configure_cpus(
        /* auto_config */ true, core, omps, util, gpus, &cpus);
    },
    ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(::testing::HasSubstr(
      fmt::format("No CPU cores left to allocate to CPU processors. Have {}, but need {} for "
                  "utility processors, and {} for GPU processors.",
                  NUM_CPUS,
                  util.value(),
                  gpus.value()))));
  ASSERT_EQ(cpus.value(), -1);
}

}  // namespace test_configure_cpus
