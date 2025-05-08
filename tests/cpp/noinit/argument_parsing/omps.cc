/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/flags/openmp.h>

#include <realm/module_config.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>
#include <vector>

namespace test_configure_omps {

class ConfigureOpenMPUnit : public DefaultFixture, public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(,
                         ConfigureOpenMPUnit,
                         ::testing::Bool(),
                         ::testing::PrintToStringParamName{});

using OpenMPsType = legate::detail::Argument<std::int32_t>;
using GPUsType    = legate::detail::Argument<std::int32_t>;

TEST_P(ConfigureOpenMPUnit, Preset)
{
  const auto gpus = GPUsType{nullptr, "--gpus", -1};
  auto omps       = OpenMPsType{nullptr, "--omps", 1};

  legate::detail::configure_omps(
    /* auto_config */ GetParam(), /* openmp */ nullptr, /* numa_mems */ {}, gpus, &omps);
  ASSERT_EQ(omps.value(), 1);
}

TEST_P(ConfigureOpenMPUnit, UnsetNoOmp)
{
  const auto gpus = GPUsType{nullptr, "--gpus", -1};
  auto omps       = OpenMPsType{nullptr, "--omps", -1};

  legate::detail::configure_omps(
    /* auto_config */ GetParam(), /* openmp */ nullptr, /* numa_mems */ {}, gpus, &omps);
  ASSERT_EQ(omps.value(), 0);
}

namespace {

class MockOpenMPModuleConfig : public Realm::ModuleConfig {
 public:
  MockOpenMPModuleConfig() : ModuleConfig{"mock_omp"} {}
};

}  // namespace

TEST_F(ConfigureOpenMPUnit, AutoConfigGPUs)
{
  constexpr auto NUM_GPUS = 128;
  const auto gpus         = GPUsType{nullptr, "--gpus", NUM_GPUS};
  const auto openmp       = MockOpenMPModuleConfig{};
  auto omps               = OpenMPsType{nullptr, "--omps", -1};

  legate::detail::configure_omps(
    /* auto_config */ true, &openmp, /* numa_mems */ {}, gpus, &omps);
  ASSERT_EQ(omps.value(), NUM_GPUS);
}

TEST_F(ConfigureOpenMPUnit, AutoConfigEmptyNUMA)
{
  const auto gpus   = GPUsType{nullptr, "--gpus", -1};
  const auto openmp = MockOpenMPModuleConfig{};
  auto omps         = OpenMPsType{nullptr, "--omps", -1};

  legate::detail::configure_omps(
    /* auto_config */ true, &openmp, /* numa_mems */ {}, gpus, &omps);
  ASSERT_EQ(omps.value(), 1);
}

TEST_F(ConfigureOpenMPUnit, NUMASize)
{
  const auto gpus      = GPUsType{nullptr, "--gpus", -1};
  const auto openmp    = MockOpenMPModuleConfig{};
  const auto numa_mems = std::vector<std::size_t>(5, 0);
  auto omps            = OpenMPsType{nullptr, "--omps", -1};

  legate::detail::configure_omps(
    /* auto_config */ true, &openmp, numa_mems, gpus, &omps);
  ASSERT_EQ(omps.value(), numa_mems.size());
}

}  // namespace test_configure_omps
