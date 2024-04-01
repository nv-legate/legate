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

#include <cstdint>

/**
 * @file
 * @brief Definition for legate::ExceptionMode
 */

namespace legate {

/**
 * @ingroup runtime
 * @brief Enum for exception handling modes
 */
enum class ExceptionMode : std::uint8_t {
  IMMEDIATE, /*!< Handles exceptions immediately. Any throwable task blocks until completion. */
  DEFERRED,  /*!< Defers all exceptions until the current scope exits. */
  IGNORED,   /*!< All exceptions are ignored. */
};

}  // namespace legate
