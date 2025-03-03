/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <cstdint>

/**
 * @file
 * @brief Definition for legate::ExceptionMode
 */

namespace legate {

/**
 * @addtogroup runtime
 * @{
 */

/**
 * @brief Enum for exception handling modes
 */
enum class ExceptionMode : std::uint8_t {
  IMMEDIATE, /*!< Handles exceptions immediately. Any throwable task blocks until completion. */
  DEFERRED,  /*!< Defers all exceptions until the current scope exits. */
  IGNORED,   /*!< All exceptions are ignored. */
};

/** @} */

}  // namespace legate
