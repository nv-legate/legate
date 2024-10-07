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

#include "legate/cuda/detail/cuda_driver_api.h"

#include "legate/utilities/detail/env.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_cuda_loader {

using CUDADriverAPITest = ::testing::Test;

TEST_F(CUDADriverAPITest, CreateDestroy)
{
  std::optional<legate::cuda::detail::CUDADriverAPI> driver{};

  // Multiple create/destroy should work (even if CUDA isn't found)
  driver.emplace();
  driver.reset();
  driver.emplace();
  driver.reset();
}

TEST_F(CUDADriverAPITest, DefaultLoadPath)
{
  // Make sure that any existing modifications are not set
  const auto _ =
    legate::test::Environment::temporary_cleared_env_var(legate::detail::LEGATE_CUDA_DRIVER);
  const legate::cuda::detail::CUDADriverAPI driver{};

  // By default, it should be looking for libcuda.so.1, because that's what is documented.
  ASSERT_THAT(driver.handle_path(), ::testing::EndsWith("libcuda.so.1"));
}

TEST_F(CUDADriverAPITest, SetLoadPath)
{
  // These are in the same test (instead of parametrized) because we want to specifically test
  // that env variable changes have immediate effect
  {
    const std::string fpath = "/this/file/does/not/exist.so";
    const auto _            = legate::test::Environment::temporary_env_var(
      legate::detail::LEGATE_CUDA_DRIVER, fpath.c_str(), true);

    const legate::cuda::detail::CUDADriverAPI driver{};

    ASSERT_EQ(driver.handle_path(), fpath);
    ASSERT_FALSE(driver.is_loaded());
    ASSERT_THROW(static_cast<void>(driver.init(0)), std::logic_error);
  }
  {
    // Test that changing the env var has immediate effect
    const std::string fpath = "/this/file/also/does/not/exist.so";
    const auto _            = legate::test::Environment::temporary_env_var(
      legate::detail::LEGATE_CUDA_DRIVER, fpath.c_str(), true);

    const legate::cuda::detail::CUDADriverAPI driver{};

    ASSERT_EQ(driver.handle_path(), fpath);
    ASSERT_FALSE(driver.is_loaded());
    ASSERT_THROW(static_cast<void>(driver.init(0)), std::logic_error);
  }
}

TEST_F(CUDADriverAPITest, TestLoad)
{
  const std::string fpath =
    LEGATE_SHARED_LIBRARY_PREFIX "legate_dummy_cuda_driver" LEGATE_SHARED_LIBRARY_SUFFIX;
  const auto _ = legate::test::Environment::temporary_env_var(
    legate::detail::LEGATE_CUDA_DRIVER, fpath.c_str(), true);

  const legate::cuda::detail::CUDADriverAPI driver{};

  ASSERT_THAT(driver.handle_path(), ::testing::EndsWith(fpath));
  ASSERT_TRUE(driver.is_loaded());
  ASSERT_EQ(driver.init(0), 0);
}

}  // namespace test_cuda_loader
