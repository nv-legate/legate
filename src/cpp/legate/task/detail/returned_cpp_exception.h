/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/return_value.h>
#include <legate/task/detail/returned_exception_common.h>
#include <legate/utilities/detail/zstring_view.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace legate::detail {

class ReturnedCppException {
 public:
  ReturnedCppException() = default;
  ReturnedCppException(std::int32_t index, std::string error_message);

  [[nodiscard]] static constexpr ExceptionKind kind();
  [[nodiscard]] std::int32_t index() const;
  [[nodiscard]] ZStringView message() const;
  [[nodiscard]] std::uint64_t size() const;
  [[nodiscard]] bool raised() const;

  [[nodiscard]] std::size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

  [[nodiscard]] ReturnValue pack() const;
  [[nodiscard]] std::string to_string() const;

  [[noreturn]] void throw_exception();

 private:
  std::int32_t index_{};
  std::string message_{};
};

}  // namespace legate::detail

#include <legate/task/detail/returned_cpp_exception.inl>
