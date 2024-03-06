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

#include "core/task/detail/return_value.h"
#include "core/task/detail/returned_cpp_exception.h"
#include "core/task/detail/returned_python_exception.h"

#include <cstddef>
#include <string>
#include <variant>

namespace legate::detail {

class ReturnedException : public std::variant<ReturnedCppException, ReturnedPythonException> {
  using base_type = std::variant<ReturnedCppException, ReturnedPythonException>;

 public:
  using base_type::base_type;

#if LEGATE_CPP_VERSION < 26
  template <typename T>
  [[nodiscard]] decltype(auto) visit(T&& fn);

  template <typename T>
  [[nodiscard]] decltype(auto) visit(T&& fn) const;
#endif

  [[nodiscard]] bool raised() const;

  [[nodiscard]] std::size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

  [[nodiscard]] ReturnValue pack() const;
  [[nodiscard]] std::string to_string() const;

  [[noreturn]] void throw_exception();

  [[nodiscard]] static ReturnedException construct_from_buffer(const void* buf);

 private:
  template <typename T>
  [[nodiscard]] static ReturnedException construct_specific_from_buffer_(const void* buf);
};

}  // namespace legate::detail

#include "core/task/detail/returned_exception.inl"
