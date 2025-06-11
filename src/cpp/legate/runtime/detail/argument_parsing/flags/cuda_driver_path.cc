/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/cuda_driver_path.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/runtime/detail/argument_parsing/argument.h>

#include <string>

namespace legate::detail {

void configure_cuda_driver_path(const Argument<std::string>& cuda_driver_path)
{
  cuda::detail::set_active_cuda_driver_api(cuda_driver_path.value());
}

}  // namespace legate::detail
