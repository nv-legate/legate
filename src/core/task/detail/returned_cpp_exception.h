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
#include "core/task/detail/returned_exception_common.h"

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

#include "core/task/detail/returned_cpp_exception.inl"
