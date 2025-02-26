/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <string_view>

namespace legate {

class ProxyArrayArgument;
class ProxyInputArguments;
class ProxyOutputArguments;
class ProxyReductionArguments;

}  // namespace legate

namespace legate::detail {

class TaskSignature;
class ProxyConstraint;

/**
 * @brief The visitor used to validate a task argument.
 */
class ValidateVisitor {
 public:
  /**
   * @brief Validate the array arguments.
   *
   * @param The specific array argument to validate.
   */
  void operator()(const ProxyArrayArgument& array) const;

  /**
   * @brief Validate the input arguments.
   *
   * Currently does nothing.
   */
  void operator()(const ProxyInputArguments&) const;

  /**
   * @brief Validate the input arguments.
   *
   * Currently does nothing.
   */
  void operator()(const ProxyOutputArguments&) const;

  /**
   * @brief Validate the array arguments.
   *
   * @param The specific array argument to validate.
   */
  void operator()(const ProxyReductionArguments&) const;

  /**
   * @brief The name of the task the signature is being validatedagainst
   */
  std::string_view task_name{};

  /**
   * @brief The task signature being validated.
   */
  const TaskSignature& signature;

  /**
   * @brief The constraint being validated.
   */
  const ProxyConstraint& constraint;
};

}  // namespace legate::detail

#include <legate/partitioning/detail/proxy/validate.inl>
