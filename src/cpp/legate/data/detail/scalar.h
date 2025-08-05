/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <string_view>

namespace legate::detail {

class Type;

class Scalar {
 public:
  // Constructs an uninitialized scalar that still owns the allocation
  // Useful for initializing stores with undefined values
  explicit Scalar(InternalSharedPtr<Type> type);
  Scalar(InternalSharedPtr<Type> type, const void* data, bool copy);
  explicit Scalar(std::string_view value);
  ~Scalar();

  template <typename T>
  explicit Scalar(T value);

  Scalar(const Scalar& other);
  Scalar(Scalar&& other) noexcept;

  Scalar& operator=(const Scalar& other);
  Scalar& operator=(Scalar&& other) noexcept;

 private:
  [[nodiscard]] static const void* copy_data_(const void* data, std::size_t size);

 public:
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;
  [[nodiscard]] const void* data() const;
  [[nodiscard]] std::size_t size() const;

  void pack(BufferBuilder& buffer) const;

 private:
  void clear_data_();

  bool own_{};
  InternalSharedPtr<Type> type_{};
  const void* data_{};
};

}  // namespace legate::detail

#include <legate/data/detail/scalar.inl>
