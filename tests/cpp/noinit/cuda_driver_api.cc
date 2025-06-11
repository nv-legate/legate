/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/cuda/detail/cuda_driver_api.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <utilities/utilities.h>

namespace test_cuda_loader {

using CUDADriverAPITest = ::testing::Test;

TEST_F(CUDADriverAPITest, CreateDestroy)
{
  std::optional<legate::cuda::detail::CUDADriverAPI> driver{};

  // Multiple create/destroy should work (even if CUDA isn't found)
  driver.emplace("foo");
  driver.reset();
  driver.emplace("bar");
  driver.reset();
}

TEST_F(CUDADriverAPITest, SetLoadPath)
{
  const std::string fpath = "/this/file/does/not/exist.so";
  const legate::cuda::detail::CUDADriverAPI driver{fpath};

  ASSERT_EQ(driver.handle_path(), fpath);
  ASSERT_FALSE(driver.is_loaded());
  ASSERT_THROW(driver.init(), std::logic_error);
}

TEST_F(CUDADriverAPITest, TestLoad)
{
  const std::string fpath =
    LEGATE_SHARED_LIBRARY_PREFIX "legate_dummy_cuda_driver" LEGATE_SHARED_LIBRARY_SUFFIX;
  const legate::cuda::detail::CUDADriverAPI driver{fpath};

  ASSERT_THAT(driver.handle_path(), ::testing::EndsWith(fpath));
  ASSERT_TRUE(driver.is_loaded());
  ASSERT_NO_THROW(driver.init());
}

}  // namespace test_cuda_loader
