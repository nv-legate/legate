/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/sysmem.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/config_realm.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/runtime/detail/argument_parsing/parse.h>

#include <realm/module_config.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utilities/utilities.h>

namespace test_configure_symem {

constexpr auto MB = 1 << 20;

class ConfigureSysMemUnit : public DefaultFixture, public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(,
                         ConfigureSysMemUnit,
                         ::testing::Bool(),
                         ::testing::PrintToStringParamName{});

using ScaledType  = legate::detail::Scaled<std::int64_t>;
using SysMemType  = legate::detail::Argument<ScaledType>;
using NUMAMemType = SysMemType;

namespace {

class MockCoreModuleConfig : public Realm::ModuleConfig {
 public:
  MockCoreModuleConfig(std::size_t sysmem_size, bool should_fail)
    : ModuleConfig{"mock_core"}, sysmem_size_{sysmem_size}
  {
    // this will make it return error REALM_MODULE_CONFIG_ERROR_NO_RESOURCE when
    // get_resource is called
    resource_discover_finished = false;
    if (!should_fail) {
      const auto inserted = resource_map.insert({"sysmem", &sysmem_size_}).second;

      LEGATE_CHECK(inserted);
      resource_discover_finished = true;
    }
  }

 private:
  std::size_t sysmem_size_{};
};

}  // namespace

TEST_P(ConfigureSysMemUnit, Preset)
{
  constexpr auto SYSMEM_SIZE = 128;
  const auto core            = MockCoreModuleConfig{1, /* should_fail */ false};
  const auto numamem         = NUMAMemType{nullptr, "--numamem", ScaledType{1, MB, "MiB"}};
  auto sysmem                = SysMemType{nullptr, "--sysmem", ScaledType{SYSMEM_SIZE, MB, "MiB"}};

  legate::detail::configure_sysmem(
    /* auto_config */ GetParam(), core, numamem, &sysmem);
  ASSERT_EQ(sysmem.value().unscaled_value(), SYSMEM_SIZE);
}

TEST_F(ConfigureSysMemUnit, NoAutoConfig)
{
  constexpr auto MINIMAL_MEM = 256;
  const auto core            = MockCoreModuleConfig{1, /* should_fail */ false};
  const auto numamem         = NUMAMemType{nullptr, "--numamem", ScaledType{0, MB, "MiB"}};
  auto sysmem                = SysMemType{nullptr, "--sysmem", ScaledType{-1, MB, "MiB"}};

  legate::detail::configure_sysmem(
    /* auto_config */ false, core, numamem, &sysmem);
  ASSERT_EQ(sysmem.value().unscaled_value(), MINIMAL_MEM);
}

TEST_F(ConfigureSysMemUnit, NUMAMemAllocated)
{
  constexpr auto MINIMAL_MEM = 256;
  const auto core            = MockCoreModuleConfig{1, /* should_fail */ false};
  const auto numamem         = NUMAMemType{nullptr, "--numamem", ScaledType{100, MB, "MiB"}};
  auto sysmem                = SysMemType{nullptr, "--sysmem", ScaledType{-1, MB, "MiB"}};

  legate::detail::configure_sysmem(
    /* auto_config */ true, core, numamem, &sysmem);
  ASSERT_EQ(sysmem.value().unscaled_value(), MINIMAL_MEM);
}

TEST_F(ConfigureSysMemUnit, AutoConfig)
{
  constexpr auto SYSMEM_SIZE = 1024;
  const auto core            = MockCoreModuleConfig{SYSMEM_SIZE * MB, /* should_fail */ false};
  const auto numamem         = NUMAMemType{nullptr, "--numamem", ScaledType{0, MB, "MiB"}};
  auto sysmem                = SysMemType{nullptr, "--sysmem", ScaledType{-1, MB, "MiB"}};

  legate::detail::configure_sysmem(
    /* auto_config */ true, core, numamem, &sysmem);

  constexpr auto EXPECTED_SYSMEM = SYSMEM_SIZE * 80 / 100;

  ASSERT_EQ(sysmem.value().unscaled_value(), EXPECTED_SYSMEM);
}

TEST_F(ConfigureSysMemUnit, AutoConfigFail)
{
  constexpr auto SYSMEM_SIZE = 1024;
  const auto core            = MockCoreModuleConfig{SYSMEM_SIZE * MB, /* should_fail */ true};
  const auto numamem         = NUMAMemType{nullptr, "--numamem", ScaledType{0, MB, "MiB"}};
  auto sysmem                = SysMemType{nullptr, "--sysmem", ScaledType{-1, MB, "MiB"}};

  ASSERT_THAT(
    [&] {
      legate::detail::configure_sysmem(
        /* auto_config */ true, core, numamem, &sysmem);
    },
    ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(
      ::testing::HasSubstr("Core Realm module could not determine the available system memory.")));
  ASSERT_EQ(sysmem.value().unscaled_value(), -1);
}

TEST_F(ConfigureSysMemUnit, RealmConfigFail)
{
  const auto parsed = legate::detail::parse_args({"parsed", "--sysmem", "1"});

  ASSERT_THAT(
    [&] { legate::detail::configure_realm(parsed); },
    ::testing::ThrowsMessage<legate::detail::ConfigurationError>(::testing::HasSubstr(
      "Unable to set core->sysmem from flag --sysmem (the Realm core module is not available).")));
}

}  // namespace test_configure_symem
