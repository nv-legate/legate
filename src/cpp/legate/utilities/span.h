/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <cuda/std/span>

#include <cstddef>

/**
 * @file
 * @brief Class definition for legate::Span.
 */

namespace legate {

/**
 * @addtogroup data
 * @{
 */

template <typename T, std::size_t N = ::cuda::std::dynamic_extent>
using Span = ::cuda::std::span<T, N>;

/** @} */

}  // namespace legate
