/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <vector>

namespace legate::detail {

/**
 * @addtogroup util
 * @{
 */

/**
 * @brief A helper class to serialize values into a contiguous buffer
 */
class BufferBuilder {
 public:
  /**
   * @brief Creates an empty buffer builder
   */
  BufferBuilder();

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
  void pack(Span<const T> values);

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

  /**
   * @brief Wraps the `BufferBuilder`'s internal allocation with a Legion `UntypedBuffer`.
   *
   * Since `UntypedBuffer` does not make a copy of the input allocation, the returned buffer
   * is good to use only as long as this buffer builder is alive.
   */
  [[nodiscard]] Legion::UntypedBuffer to_legion_buffer() const;

 private:
  std::vector<std::int8_t> buffer_{};
};

/** @} */

}  // namespace legate::detail

#include <legate/utilities/detail/buffer_builder.inl>
