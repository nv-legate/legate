/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <cuda/std/complex>

namespace legate {

/**
 * @addtogroup types
 * @{
 */

/** @file */

/**
 * @brief The complex type used by Legate.
 *
 * @tparam T The inner type for the real and imaginary parts. Usually this is a floating-point
 * type such as `float` or `double`.
 *
 * See
 * [cuda::std::complex](https://nvidia.github.io/cccl/libcudacxx/standard_api/numerics_library/complex.html)
 * for a full description of the class and its members.
 */
template <typename T>
using Complex = ::cuda::std::complex<T>;

/** @} */

}  // namespace legate
