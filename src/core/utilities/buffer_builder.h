/* Copyright 2021-2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "legion.h"

#include "core/utilities/tuple.h"

namespace legate {

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
   * @param buffer Buffer to serialize
   * @param size Size of the buffer
   */
  void pack_buffer(const void* buffer, size_t size);

 public:
  /**
   * @brief Wraps the `BufferBuilder`'s internal allocation with a Legion `UntypedBuffer`.
   *
   * Since `UntypedBuffer` does not make a copy of the input allocation, the returned buffer
   * is good to use only as long as this buffer builder is alive.
   */
  Legion::UntypedBuffer to_legion_buffer() const;

 private:
  std::vector<int8_t> buffer_;
};

}  // namespace legate

#include "core/utilities/buffer_builder.inl"
