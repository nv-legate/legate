/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define LEGATE_CUDA_DRIVER_API_MOCK 1

// Must come first because we need the FRIEND_TEST() macros to work in cuda_driver_api.h
#include <gmock/gmock.h>
#include <gtest/gtest.h>
//

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/runtime/detail/argument_parsing/flags/fbmem.h>

#include <realm/module_config.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace {

constexpr auto MB = 1 << 20;

}

namespace test_configure_fbmem {

class ConfigureFBMemUnit : public DefaultFixture, public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(,
                         ConfigureFBMemUnit,
                         ::testing::Bool(),
                         ::testing::PrintToStringParamName{});

using ScaledType = legate::detail::Scaled<std::int64_t>;
using FBMemType  = legate::detail::Argument<ScaledType>;
using GPUsType   = legate::detail::Argument<std::int32_t>;

TEST_P(ConfigureFBMemUnit, Preset)
{
  constexpr auto FBMEM_SIZE = 128;
  auto fbmem                = FBMemType{nullptr, "--fbmem", ScaledType{FBMEM_SIZE, MB, "MiB"}};
  const auto gpus           = GPUsType{nullptr, "--gpus", 0};

  legate::detail::configure_fbmem(/* auto_config */ GetParam(), /* cuda */ nullptr, gpus, &fbmem);
  ASSERT_EQ(fbmem.value().unscaled_value(), FBMEM_SIZE);
}

TEST_P(ConfigureFBMemUnit, UnsetNoGPUs)
{
  auto fbmem      = FBMemType{nullptr, "--fbmem", ScaledType{-1, MB, "MiB"}};
  const auto gpus = GPUsType{nullptr, "--gpus", 0};

  legate::detail::configure_fbmem(/* auto_config */ GetParam(), /* cuda */ nullptr, gpus, &fbmem);
  ASSERT_EQ(fbmem.value().unscaled_value(), 0);
}

TEST_P(ConfigureFBMemUnit, UnsetNoCUDA)
{
  constexpr auto MINIMAL_MEM = 256;
  auto fbmem                 = FBMemType{nullptr, "--fbmem", ScaledType{-1, MB, "MiB"}};
  const auto gpus            = GPUsType{nullptr, "--gpus", 1};

  legate::detail::configure_fbmem(/* auto_config */ GetParam(), /* cuda */ nullptr, gpus, &fbmem);
  ASSERT_EQ(fbmem.value().unscaled_value(), MINIMAL_MEM);
}

}  // namespace test_configure_fbmem

// These tests need to be in namespace legate otherwise the FRIEND_TEST() macros don't work

namespace legate::cuda::detail {

using ::test_configure_fbmem::ConfigureFBMemUnit;
using ::test_configure_fbmem::FBMemType;
using ::test_configure_fbmem::GPUsType;
using ::test_configure_fbmem::ScaledType;

namespace {

class MockCUDAModuleConfig : public Realm::ModuleConfig {
 public:
  MockCUDAModuleConfig() : ModuleConfig{"mock_cuda"} {}
};

class AutoDriverAPIMockBase {
 protected:
  AutoDriverAPIMockBase()
    : api_{[] {
        legate::cuda::detail::set_active_cuda_driver_api("/nonexistent/cuda/driver.so");
        return legate::cuda::detail::get_cuda_driver_api();
      }()},
      prev_{std::move(*api_)}
  {
  }

  virtual ~AutoDriverAPIMockBase() { *api_ = std::move(prev_); }

  InternalSharedPtr<CUDADriverAPI> api_{};
  CUDADriverAPI prev_;
};

}  // namespace

TEST_F(ConfigureFBMemUnit, AutoConfigCUDA)
{
  static constexpr std::size_t FBMEM_SIZE = 256;

  // We only need to mock the functions required to do:
  //
  // 1. AutoPrimaryContext:
  //    - device_primary_ctx_retain
  //    - ctx_push_current
  //    - ctx_pop_current
  //    - device_primary_ctx_release
  // 2. mem_get_info
  class AutoDriverAPIMock : public AutoDriverAPIMockBase {
   public:
    AutoDriverAPIMock()
    {
      // Can be anything, so long as it loads (to defeat the "are we initialized" checks). We
      // use nullptr to load the current shared library, which is -- by definition --
      // guaranteed to exist.
      api_->lib_                        = legate::detail::SharedLibrary{nullptr};
      api_->device_primary_ctx_retain_  = mock_device_primary_ctx_retain_;
      api_->ctx_push_current_           = mock_ctx_push_current_;
      api_->ctx_pop_current_            = mock_ctx_pop_current_;
      api_->device_primary_ctx_release_ = mock_device_primary_ctx_release_;
      api_->mem_get_info_               = mock_mem_get_info_;
    }

   private:
    static CUresult mock_device_primary_ctx_retain_(CUcontext* ctx, CUdevice)
    {
      *ctx = nullptr;
      return 0;
    }

    static CUresult mock_ctx_push_current_(CUcontext) { return 0; }

    static CUresult mock_ctx_pop_current_(CUcontext* ctx)
    {
      *ctx = nullptr;
      return 0;
    }

    static CUresult mock_device_primary_ctx_release_(CUdevice) { return 0; }

    static CUresult mock_mem_get_info_(std::size_t* fbmem_size, std::size_t*)
    {
      *fbmem_size = FBMEM_SIZE * MB;
      return 0;
    }
  };

  auto fbmem      = FBMemType{nullptr, "--fbmem", ScaledType{-1, MB, "MiB"}};
  const auto gpus = GPUsType{nullptr, "--gpus", 1};
  auto cuda       = MockCUDAModuleConfig{};

  {
    const auto _ = AutoDriverAPIMock{};

    legate::detail::configure_fbmem(/* auto_config */ true, &cuda, gpus, &fbmem);
  }

  constexpr auto RES_FBMEM = FBMEM_SIZE * 95 / 100;

  ASSERT_EQ(fbmem.value().unscaled_value(), RES_FBMEM);
}

TEST_F(ConfigureFBMemUnit, AutoConfigFail)
{
  class AutoDriverAPIMock : public AutoDriverAPIMockBase {
   public:
    AutoDriverAPIMock()
    {
      // Can be anything, so long as it loads (to defeat the "are we initialized" checks). We
      // use nullptr to load the current shared library, which is -- by definition --
      // guaranteed to exist.
      api_->lib_                       = legate::detail::SharedLibrary{nullptr};
      api_->device_primary_ctx_retain_ = [](CUcontext*, CUdevice) -> CUresult {
        throw std::exception{};
      };
    }
  };

  auto fbmem      = FBMemType{nullptr, "--fbmem", ScaledType{-1, MB, "MiB"}};
  const auto gpus = GPUsType{nullptr, "--gpus", 1};
  auto cuda       = MockCUDAModuleConfig{};

  ASSERT_THAT(
    [&] {
      const auto _ = AutoDriverAPIMock{};

      legate::detail::configure_fbmem(/* auto_config */ true, &cuda, gpus, &fbmem);
    },
    ::testing::ThrowsMessage<legate::detail::AutoConfigurationError>(
      ::testing::HasSubstr("Unable to determine the available GPU memory.")));
  ASSERT_EQ(fbmem.value().unscaled_value(), -1);
}

}  // namespace legate::cuda::detail
