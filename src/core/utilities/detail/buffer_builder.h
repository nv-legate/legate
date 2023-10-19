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

#include "legion.h"

#include "core/utilities/tuple.h"

namespace legate::detail {

/**
 * @ingroup util
 * @brief A helper class to serialize values into a contiguous buffer
 */
class BufferBuilder {
 public:
  /**
   * @brief Creates an empty buffer builder
   */
  BufferBuilder();

 public:
  /**
   * @brief Serializes a value
   *
   * @param value Value to serialize
   */
  template <typename T>
  void pack(const T& value);
  /**
   * @brief Serializes multiple values
   *
   * @param values Values to serialize in a vector
   */
  template <typename T>
  void pack(const std::vector<T>& values);
  /**
   * @brief Serializes multiple values
   *
   * @param values Values to serialize in a tuple
   */
  template <typename T>
  void pack(const tuple<T>& values);
  /**
   * @brief Serializes an arbitrary allocation
   *
   * The caller should make sure that `(char*)buffer + (size - 1)` is a valid address.
   *
   * @param mem Buffer to serialize
   * @param size Size of the buffer
   * @param align Alignment of mem (must be a power of 2)
   */
  void pack_buffer(const void* mem, std::size_t size, std::size_t align);

 public:
  /**
   * @brief Wraps the `BufferBuilder`'s internal allocation with a Legion `UntypedBuffer`.
   *
   * Since `UntypedBuffer` does not make a copy of the input allocation, the returned buffer
   * is good to use only as long as this buffer builder is alive.
   */
  Legion::UntypedBuffer to_legion_buffer() const;

 public:
  BufferBuilder(const BufferBuilder&)            = default;
  BufferBuilder& operator=(const BufferBuilder&) = default;
  BufferBuilder(BufferBuilder&&)                 = default;
  BufferBuilder& operator=(BufferBuilder&&)      = default;

 private:
  std::vector<int8_t> buffer_;
};

}  // namespace legate::detail

#include "core/utilities/detail/buffer_builder.inl"
