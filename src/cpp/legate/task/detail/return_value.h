/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/typedefs.h>

#include <cstddef>

namespace legate::detail {

class ReturnValue {
 public:
  ReturnValue(Legion::UntypedDeferredValue value, std::size_t size, std::size_t alignment);

  ReturnValue(const ReturnValue&)            = default;
  ReturnValue& operator=(const ReturnValue&) = default;

  [[nodiscard]] void* ptr();
  [[nodiscard]] const void* ptr() const;
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] std::size_t alignment() const;
  [[nodiscard]] bool is_device_value() const;

  // Calls the Legion postamble with an instance
  void finalize(Legion::Context legion_context) const;

 private:
  Legion::UntypedDeferredValue value_{};
  std::size_t size_{};
  std::size_t alignment_{};
  bool is_device_value_{};
};

}  // namespace legate::detail

#include <legate/task/detail/return_value.inl>
