/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/gpus.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/runtime/detail/config.h>

#include <realm/module_config.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace test_configure_gpus {

class ConfigureGPUsUnit : public DefaultFixture, public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(,
                         ConfigureGPUsUnit,
                         ::testing::Bool(),
                         ::testing::PrintToStringParamName{});

TEST_P(ConfigureGPUsUnit, Preset)
{
  constexpr auto NUM_GPUS = 44;
  auto arg                = legate::detail::Argument<std::int32_t>{nullptr, "--gpus", NUM_GPUS};
  auto cfg                = legate::detail::Config{};

  cfg.set_need_cuda(false);
  legate::detail::configure_gpus(/* auto_config */ GetParam(), /* cuda */ nullptr, &arg, &cfg);
  ASSERT_EQ(arg.value(), NUM_GPUS);
  ASSERT_TRUE(cfg.need_cuda());
}

namespace {

class MockCUDAModuleConfig : public Realm::ModuleConfig {
 public:
  explicit MockCUDAModuleConfig(std::int32_t num_gpus, bool should_fail)
    : ModuleConfig{"mock_cuda"}, num_gpus_{num_gpus}
  {
    // this will make it return error REALM_MODULE_CONFIG_ERROR_NO_RESOURCE when
    // get_resource is called
    resource_discover_finished = false;
    if (!should_fail) {
      const auto inserted = resource_map.insert({"gpu", &num_gpus_}).second;

      LEGATE_CHECK(inserted);
      resource_discover_finished = true;
    }
  }

 private:
  std::int32_t num_gpus_{};
};

}  // namespace

TEST_F(ConfigureGPUsUnit, UnsetNoAutoConfigure)
{
  constexpr auto NUM_GPUS = 123;
  auto arg                = legate::detail::Argument<std::int32_t>{nullptr, "--gpus", -1};
  auto cfg                = legate::detail::Config{};
  auto cuda               = MockCUDAModuleConfig{NUM_GPUS, /* should_fail */ false};

  legate::detail::configure_gpus(/* auto_config */ false, /* cuda */ &cuda, &arg, &cfg);
  ASSERT_EQ(arg.value(), 0);
  ASSERT_FALSE(cfg.need_cuda());
}

TEST_F(ConfigureGPUsUnit, UnsetAutoConfigure)
{
  constexpr auto NUM_GPUS = 123;
  auto arg                = legate::detail::Argument<std::int32_t>{nullptr, "--gpus", -1};
  auto cfg                = legate::detail::Config{};
  auto cuda               = MockCUDAModuleConfig{NUM_GPUS, /* should_fail */ false};

  cfg.set_need_cuda(false);
  legate::detail::configure_gpus(/* auto_config */ true, &cuda, &arg, &cfg);
  ASSERT_EQ(arg.value(), NUM_GPUS);
  ASSERT_TRUE(cfg.need_cuda());
}

TEST_F(ConfigureGPUsUnit, UnsetAutoConfigureNoCUDA)
{
  auto arg = legate::detail::Argument<std::int32_t>{nullptr, "--gpus", -1};
  auto cfg = legate::detail::Config{};

  legate::detail::configure_gpus(/* auto_config */ true, /* cuda */ nullptr, &arg, &cfg);
  ASSERT_EQ(arg.value(), 0);
  ASSERT_FALSE(cfg.need_cuda());
}

TEST_F(ConfigureGPUsUnit, CUDAResourceFail)
{
  auto arg  = legate::detail::Argument<std::int32_t>{nullptr, "--gpus", -1};
  auto cfg  = legate::detail::Config{};
  auto cuda = MockCUDAModuleConfig{0, /* should_fail */ true};

  ASSERT_THAT([&] { legate::detail::configure_gpus(/* auto_config */ true, &cuda, &arg, &cfg); },
              ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(
                ::testing::HasSubstr("CUDA Realm module could not determine the number of GPUs.")));
  ASSERT_EQ(arg.value(), -1);
  ASSERT_FALSE(cfg.need_cuda());
}

}  // namespace test_configure_gpus
