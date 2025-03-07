/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/return_value.h>
#include <legate/task/detail/returned_exception_common.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace legate::detail {

class ReturnedPythonException {
 public:
  constexpr ReturnedPythonException() = default;

  /**
   * @brief Construct a Python exception.
   *
   * @param pkl_buf A pointer to the bytes contained the pickled exception instance.
   * @param pkl_len The size of the pickle buffer.
   * @param msg A string holding the formatted textual representation of the exception.
   *
   * This ctor exists purely because Cython doesn't know how to create `Span`'s. See the other
   * ctor for further info.
   */
  ReturnedPythonException(const std::byte* pkl_buf, std::size_t pkl_len, std::string msg);

  /**
   * @brief Construct a Python exception.
   *
   * @param pkl_span The bytes contained the pickled exception instance.
   * @param msg A string holding the formatted textual representation of the exception.
   *
   * `message` holds the contents of
   *
   * @code{.py}
   * msg = traceback.format_exn(exception)
   * @endcode
   *
   * It will contain the raw-text traceback output that you get whenever a Python exception
   * reaches the top, i.e.
   *
   * @code{.bash}
   * $ python3 -c 'raise RuntimeError()'
   * Traceback (most recent call last):
   *   File "<string>", line 1, in <module>
   * RuntimeError
   * @endcode
   *
   * `pkl_bytes` holds the contents of
   *
   * @code{.py}
   * bytes = pickle.dumps(exception)
   * @endcode
   *
   * It will contain the encoded bytes that Python will use to reconstruct the entire
   * exception. Technically it _does_ also contain the contents of message (very likely almost
   * verbatim), but being able to extract that in C++ (and save ourselves from storing
   * `message`) requires us to recreate the pickle protocol.
   *
   * So it's easier to just store both. It's inefficient, sure, but we only do this when we are
   * handling an uncaught exception, and throwing exceptions was never going to be cheap anyways.
   */
  ReturnedPythonException(Span<const std::byte> pkl_span, std::string msg);

  [[nodiscard]] static constexpr ExceptionKind kind();
  [[nodiscard]] Span<const std::byte> pickle() const;
  [[nodiscard]] std::string_view message() const;
  [[nodiscard]] bool raised() const;

  [[nodiscard]] std::size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

  [[nodiscard]] ReturnValue pack() const;
  [[nodiscard]] std::string to_string() const;

  [[noreturn]] void throw_exception();

 private:
  class Payload {
   public:
    Payload() = default;
    Payload(std::size_t size, std::unique_ptr<std::byte[]> bytes, std::string m) noexcept;

    std::size_t pkl_size{};
    std::unique_ptr<std::byte[]> pkl_bytes{};
    std::string msg{};
  };

  InternalSharedPtr<Payload> bytes_{};
};

}  // namespace legate::detail

#include <legate/task/detail/returned_python_exception.inl>
