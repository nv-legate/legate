/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/ompthreads.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/runtime/detail/config.h>

#include <realm/module_config.h>

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utilities/utilities.h>

namespace test_configure_ompthreads {

class ConfigureOpenMPThreadsUnit : public DefaultFixture,
                                   public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(,
                         ConfigureOpenMPThreadsUnit,
                         ::testing::Bool(),
                         ::testing::PrintToStringParamName{});

using OpenMPsType       = legate::detail::Argument<std::int32_t>;
using UtilType          = OpenMPsType;
using CPUsType          = OpenMPsType;
using GPUsType          = OpenMPsType;
using OpenMPsType       = OpenMPsType;
using OpenMPThreadsType = OpenMPsType;

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

TEST_P(ConfigureOpenMPThreadsUnit, Preset)
{
  constexpr auto NUM_OMPTHREADS = 10;
  const auto core               = MockCoreModuleConfig{/*num_cpus=*/8, /* should_fail */ false};
  const auto util               = UtilType{nullptr, "--util", /*init=*/1};
  const auto cpus               = CPUsType{nullptr, "--cpus", /*init=*/2};
  const auto gpus               = GPUsType{nullptr, "--gpus", /*init=*/3};
  const auto omps               = OpenMPsType{nullptr, "--omps", /*init=*/4};
  auto ompthreads               = OpenMPThreadsType{nullptr, "--ompthreads", NUM_OMPTHREADS};
  auto cfg                      = legate::detail::Config{};

  cfg.set_need_openmp(false);
  cfg.set_num_omp_threads(0);
  legate::detail::configure_ompthreads(
    /* auto_config */ GetParam(), core, util, cpus, gpus, omps, &ompthreads, &cfg);
  ASSERT_EQ(ompthreads.value(), NUM_OMPTHREADS);
  ASSERT_TRUE(cfg.need_openmp());
  ASSERT_EQ(cfg.num_omp_threads(), NUM_OMPTHREADS);
}

TEST_P(ConfigureOpenMPThreadsUnit, NoOpenMP)
{
  const auto core = MockCoreModuleConfig{/*num_cpus=*/8, /* should_fail */ false};
  const auto util = UtilType{nullptr, "--util", /*init=*/1};
  const auto cpus = CPUsType{nullptr, "--cpus", /*init=*/2};
  const auto gpus = GPUsType{nullptr, "--gpus", /*init=*/3};
  const auto omps = OpenMPsType{nullptr, "--omps", /*init=*/0};
  auto ompthreads = OpenMPThreadsType{nullptr, "--ompthreads", /*init=*/-1};
  auto cfg        = legate::detail::Config{};

  legate::detail::configure_ompthreads(
    /* auto_config */ GetParam(), core, util, cpus, gpus, omps, &ompthreads, &cfg);
  ASSERT_EQ(ompthreads.value(), 0);
  ASSERT_FALSE(cfg.need_openmp());
  ASSERT_EQ(cfg.num_omp_threads(), 0);
}

TEST_F(ConfigureOpenMPThreadsUnit, NoAutoConfig)
{
  const auto core = MockCoreModuleConfig{/*num_cpus=*/8, /* should_fail */ false};
  const auto util = UtilType{nullptr, "--util", /*init=*/1};
  const auto cpus = CPUsType{nullptr, "--cpus", /*init=*/2};
  const auto gpus = GPUsType{nullptr, "--gpus", /*init=*/3};
  const auto omps = OpenMPsType{nullptr, "--omps", /*init=*/4};
  auto ompthreads = OpenMPThreadsType{nullptr, "--ompthreads", /*init=*/-1};
  auto cfg        = legate::detail::Config{};

  legate::detail::configure_ompthreads(
    /* auto_config */ false, core, util, cpus, gpus, omps, &ompthreads, &cfg);
  ASSERT_EQ(ompthreads.value(), 1);
  ASSERT_TRUE(cfg.need_openmp());
  ASSERT_EQ(cfg.num_omp_threads(), 1);
}

TEST_F(ConfigureOpenMPThreadsUnit, AutoConfig)
{
  constexpr auto NUM_CPUS = 16;
  const auto core         = MockCoreModuleConfig{NUM_CPUS, /* should_fail */ false};
  const auto util         = UtilType{nullptr, "--util", /*init=*/1};
  const auto cpus         = CPUsType{nullptr, "--cpus", /*init=*/2};
  const auto gpus         = GPUsType{nullptr, "--gpus", /*init=*/3};
  const auto omps         = OpenMPsType{nullptr, "--omps", /*init=*/4};
  auto ompthreads         = OpenMPThreadsType{nullptr, "--ompthreads", /*init=*/-1};
  auto cfg                = legate::detail::Config{};

  legate::detail::configure_ompthreads(
    /* auto_config */ true, core, util, cpus, gpus, omps, &ompthreads, &cfg);

  const auto expected_ompthreads = static_cast<std::int32_t>(
    std::floor((NUM_CPUS - cpus.value() - util.value() - gpus.value()) / omps.value()));

  ASSERT_EQ(ompthreads.value(), expected_ompthreads);
  ASSERT_TRUE(cfg.need_openmp());
  ASSERT_EQ(cfg.num_omp_threads(), expected_ompthreads);
}

TEST_F(ConfigureOpenMPThreadsUnit, AutoConfigFail)
{
  const auto core = MockCoreModuleConfig{/*num_cpus=*/1, /* should_fail */ true};
  const auto util = UtilType{nullptr, "--util", /*init=*/1};
  const auto cpus = CPUsType{nullptr, "--cpus", /*init=*/2};
  const auto gpus = GPUsType{nullptr, "--gpus", /*init=*/3};
  const auto omps = OpenMPsType{nullptr, "--omps", /*init=*/4};
  auto ompthreads = OpenMPThreadsType{nullptr, "--ompthreads", /*init=*/-1};
  auto cfg        = legate::detail::Config{};

  ASSERT_THAT(
    [&] {
      legate::detail::configure_ompthreads(
        /* auto_config */ true, core, util, cpus, gpus, omps, &ompthreads, &cfg);
    },
    ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(
      ::testing::HasSubstr("Core Realm module could not determine the number of CPU cores.")));
  ASSERT_EQ(ompthreads.value(), -1);
  ASSERT_FALSE(cfg.need_openmp());
  ASSERT_EQ(cfg.num_omp_threads(), 0);
}

TEST_F(ConfigureOpenMPThreadsUnit, AutoConfigFailNoCPUs)
{
  const auto core = MockCoreModuleConfig{/*num_cpus=*/0, /* should_fail */ false};
  const auto util = UtilType{nullptr, "--util", /*init=*/1};
  const auto cpus = CPUsType{nullptr, "--cpus", /*init=*/2};
  const auto gpus = GPUsType{nullptr, "--gpus", /*init=*/3};
  const auto omps = OpenMPsType{nullptr, "--omps", /*init=*/4};
  auto ompthreads = OpenMPThreadsType{nullptr, "--ompthreads", /*init=*/-1};
  auto cfg        = legate::detail::Config{};

  ASSERT_THAT(
    [&] {
      legate::detail::configure_ompthreads(
        /* auto_config */ true, core, util, cpus, gpus, omps, &ompthreads, &cfg);
    },
    ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(::testing::HasSubstr(
      "Core Realm module detected 0 CPU cores while configuring the number of OpenMP threads.")));
  ASSERT_EQ(ompthreads.value(), -1);
  ASSERT_FALSE(cfg.need_openmp());
  ASSERT_EQ(cfg.num_omp_threads(), 0);
}

TEST_F(ConfigureOpenMPThreadsUnit, AutoConfigFailNotEnoughOMPThreads)
{
  constexpr auto NUM_CPUS = 1;
  const auto core         = MockCoreModuleConfig{NUM_CPUS, /* should_fail */ false};
  const auto util         = UtilType{nullptr, "--util", /*init=*/1};
  const auto cpus         = CPUsType{nullptr, "--cpus", /*init=*/2};
  const auto gpus         = GPUsType{nullptr, "--gpus", /*init=*/3};
  const auto omps         = OpenMPsType{nullptr, "--omps", /*init=*/4};
  auto ompthreads         = OpenMPThreadsType{nullptr, "--ompthreads", /*init=*/-1};
  auto cfg                = legate::detail::Config{};

  ASSERT_THAT(
    [&] {
      legate::detail::configure_ompthreads(
        /* auto_config */ true, core, util, cpus, gpus, omps, &ompthreads, &cfg);
    },
    ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(::testing::HasSubstr(
      fmt::format("Not enough CPU cores to split across {} OpenMP processor(s). Have {}, but need "
                  "{} for CPU processors, {} for utility processors, {} for GPU processors, and at "
                  "least {} for OpenMP processors (1 core each).",
                  omps.value(),
                  NUM_CPUS,
                  cpus.value(),
                  util.value(),
                  gpus.value(),
                  omps.value()))));
  ASSERT_EQ(ompthreads.value(), -1);
  ASSERT_FALSE(cfg.need_openmp());
  ASSERT_EQ(cfg.num_omp_threads(), 0);
}

}  // namespace test_configure_ompthreads
