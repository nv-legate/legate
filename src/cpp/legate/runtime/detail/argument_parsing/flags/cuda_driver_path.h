/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace legate::detail {

template <typename T>
class Argument;

void configure_cuda_driver_path(const Argument<std::string>& cuda_driver_path);

}  // namespace legate::detail
