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

#include "core/task/exception.h"
#include "core/utilities/typedefs.h"

#include <array>
#include <optional>
#include <string_view>
#include <vector>

namespace legate::detail {

struct ReturnValue {
 public:
  ReturnValue(Legion::UntypedDeferredValue value, size_t size);

  ReturnValue(const ReturnValue&)            = default;
  ReturnValue& operator=(const ReturnValue&) = default;

  [[nodiscard]] static ReturnValue unpack(const void* ptr, size_t size, Memory::Kind memory_kind);

  [[nodiscard]] void* ptr();
  [[nodiscard]] const void* ptr() const;
  [[nodiscard]] size_t size() const;
  [[nodiscard]] bool is_device_value() const;

  // Calls the Legion postamble with an instance
  void finalize(Legion::Context legion_context) const;

 private:
  Legion::UntypedDeferredValue value_{};
  size_t size_{};
  bool is_device_value_{};
};

struct ReturnedException {
 public:
  ReturnedException() = default;
  ReturnedException(int32_t index, std::string_view error_message);

  static inline constexpr auto MAX_MESSAGE_SIZE = 256;

  [[nodiscard]] bool raised() const;

  [[nodiscard]] std::optional<TaskException> to_task_exception() const;

  [[nodiscard]] size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

  [[nodiscard]] ReturnValue pack() const;

 private:
  bool raised_{};
  int32_t index_{-1};
  uint32_t message_size_{};
  std::array<char, MAX_MESSAGE_SIZE> error_message_{};
};

struct ReturnValues {
 public:
  ReturnValues() = default;
  ReturnValues(std::vector<ReturnValue>&& return_values);

  [[nodiscard]] ReturnValue operator[](int32_t idx) const;

  [[nodiscard]] size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

  [[nodiscard]] static ReturnValue extract(const Legion::Future& future, uint32_t to_extract);

  // Calls the Legion postamble with an instance that packs all return values
  void finalize(Legion::Context legion_context) const;

 private:
  size_t buffer_size_{};
  std::vector<ReturnValue> return_values_{};
};

}  // namespace legate::detail

#include "core/task/detail/return.inl"
