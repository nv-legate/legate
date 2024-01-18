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

#include "core/type/detail/type_info.h"
#include "core/utilities/detail/buffer_builder.h"
#include "core/utilities/internal_shared_ptr.h"

namespace legate::detail {

class Type;

class Scalar {
 public:
  // Constructs an uninitialized scalar that still owns the allocation
  // Useful for initializing stores with undefined values
  explicit Scalar(InternalSharedPtr<Type> type);
  Scalar(InternalSharedPtr<Type> type, const void* data, bool copy);
  explicit Scalar(const std::string& value);
  ~Scalar();

  template <typename T>
  explicit Scalar(T value);

  Scalar(const Scalar& other);
  Scalar(Scalar&& other) noexcept;

  Scalar& operator=(const Scalar& other);
  Scalar& operator=(Scalar&& other) noexcept;

 private:
  [[nodiscard]] static const void* copy_data(const void* data, size_t size);

 public:
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;
  [[nodiscard]] const void* data() const;
  [[nodiscard]] size_t size() const;

  void pack(BufferBuilder& buffer) const;

 private:
  void clear_data();

  bool own_{};
  InternalSharedPtr<Type> type_{};
  const void* data_{};
};

}  // namespace legate::detail

#include "core/data/detail/scalar.inl"
