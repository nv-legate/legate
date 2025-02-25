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

namespace legate::proxy {

class ArrayArgument;
class InputArguments;
class OutputArguments;
class ReductionArguments;

}  // namespace legate::proxy

namespace legate::detail {

class TaskSignature;

}  // namespace legate::detail

namespace legate::detail::proxy {

class Constraint;

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
  void operator()(const legate::proxy::ArrayArgument& array) const;

  /**
   * @brief Validate the input arguments.
   *
   * Currently does nothing.
   */
  void operator()(const legate::proxy::InputArguments&) const;

  /**
   * @brief Validate the input arguments.
   *
   * Currently does nothing.
   */
  void operator()(const legate::proxy::OutputArguments&) const;

  /**
   * @brief Validate the array arguments.
   *
   * @param The specific array argument to validate.
   */
  void operator()(const legate::proxy::ReductionArguments&) const;

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
  const Constraint& constraint;
};

}  // namespace legate::detail::proxy

#include <legate/partitioning/detail/proxy/validate.inl>
