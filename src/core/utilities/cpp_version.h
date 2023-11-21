/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#ifdef __cplusplus
#if __cplusplus <= 199711L
#ifdef _MSC_VER
// Unless a special flag is set, MSVC always reports C++ standard as C++98. But the floor is in
// fact C++14, so we assume that that is the case.
// See https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus?view=msvc-170#remarks
#define LEGATE_CPP_VERSION 14
#else
// wrap C++98 to 0 since comparisons would otherwise fail
#define LEGATE_CPP_VERSION 0
#endif
#elif __cplusplus <= 201103L
#define LEGATE_CPP_VERSION 11
#elif __cplusplus <= 201402L
#define LEGATE_CPP_VERSION 14
#elif __cplusplus <= 201703L
#define LEGATE_CPP_VERSION 17
#elif __cplusplus <= 202002L
#define LEGATE_CPP_VERSION 20
#else
#define LEGATE_CPP_VERSION 23  // current year, or date of c++2b ratification
#endif
#else
#define LEGATE_CPP_VERSION 0  // no C++
#endif                        // __cplusplus

#define LEGATE_CPP_MIN_VERSION 17
#if defined(__cplusplus) && LEGATE_CPP_VERSION < LEGATE_CPP_MIN_VERSION
#error "Legate requires C++" #LEGATE_CPP_MIN_VERSION
#endif
