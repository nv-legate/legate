/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/cuda_driver_path.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/runtime/detail/argument_parsing/argument.h>

#include <gtest/gtest.h>

#include <string>
#include <utilities/utilities.h>

namespace test_configure_cuda_driver_path {

class ConfigureCUDADriverPathUnit : public DefaultFixture {};

using DriverPathType = legate::detail::Argument<std::string>;

TEST_F(ConfigureCUDADriverPathUnit, Basic)
{
  const auto lib_name = std::string{"/path/to/libfoo.so"};
  auto path           = DriverPathType{nullptr, "--cuda-driver-path", lib_name};

  legate::detail::configure_cuda_driver_path(path);

  auto&& api = legate::cuda::detail::get_cuda_driver_api();

  ASSERT_EQ(api->handle_path(), lib_name);
}

}  // namespace test_configure_cuda_driver_path
