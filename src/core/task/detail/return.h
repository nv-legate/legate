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

#include <optional>
#include <vector>
#include "core/task/exception.h"
#include "core/utilities/typedefs.h"

namespace legate::detail {

struct ReturnValue {
 public:
  ReturnValue(Legion::UntypedDeferredValue value, size_t size);

 public:
  ReturnValue(const ReturnValue&)            = default;
  ReturnValue& operator=(const ReturnValue&) = default;

 public:
  static ReturnValue unpack(const void* ptr, size_t size, Memory::Kind memory_kind);

 public:
  void* ptr();
  const void* ptr() const;
  size_t size() const { return size_; }
  bool is_device_value() const { return is_device_value_; }

 public:
  // Calls the Legion postamble with an instance
  void finalize(Legion::Context legion_context) const;

 private:
  Legion::UntypedDeferredValue value_{};
  size_t size_{0};
  bool is_device_value_{false};
};

struct ReturnedException {
 public:
  ReturnedException() {}
  ReturnedException(int32_t index, const std::string& error_message);

 public:
  bool raised() const { return raised_; }

 public:
  std::optional<TaskException> to_task_exception() const;

 public:
  size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

 public:
  ReturnValue pack() const;

 private:
  bool raised_{false};
  int32_t index_{-1};
  std::string error_message_{};
};

struct ReturnValues {
 public:
  ReturnValues();
  ReturnValues(std::vector<ReturnValue>&& return_values);

 public:
  ReturnValues(const ReturnValues&)            = default;
  ReturnValues& operator=(const ReturnValues&) = default;

 public:
  ReturnValues(ReturnValues&&)            = default;
  ReturnValues& operator=(ReturnValues&&) = default;

 public:
  ReturnValue operator[](int32_t idx) const;

 public:
  size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

 public:
  static ReturnValue extract(Legion::Future future, uint32_t to_extract);

 public:
  // Calls the Legion postamble with an instance that packs all return values
  void finalize(Legion::Context legion_context) const;

 private:
  size_t buffer_size_{0};
  std::vector<ReturnValue> return_values_{};
};

}  // namespace legate::detail
