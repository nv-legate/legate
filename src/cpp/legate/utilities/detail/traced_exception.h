/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <new>
#include <string_view>

namespace legate::detail {

/**
 * @brief Install the Legate `std::terminate()` handler.
 *
 * @return `true` if the handlers were installed, `false` otherwise.
 *
 * This routine is thread-safe, and may be called multiple times. However, only the *first*
 * invocation has any effect. Subsequent calls to this function have no effect. The user may
 * respect the return value to determine whether the handler was installed.
 *
 * The installed handler will pretty-print any thrown exceptions, adding a traceback showing
 * where the exception was thrown.
 */
[[nodiscard]] bool install_terminate_handler() noexcept;

/**
 * @brief The base class for traced exceptions.
 *
 * This class exists to serve as a type-erased target in "catch" clauses aiming to only catch
 * traced exceptions.
 */
class LEGATE_EXPORT TracedExceptionBase {
  class Impl;

 public:
  /**
   * @brief Construct a TracedExceptionBase.
   *
   * @param ptr An exception pointer to the original exception object.
   * @param skip_frames The number of stacktrace frames to skip.
   *
   * `TracedExceptionBase` will add additional skip frames for each constructor/function-call
   * made by it, so the user should not try to account for that. The user should only try to
   * account for its immediate frames. So if a `TracedExceptionBase` object is constructed via
   * the path `interesting() -> foo() -> bar() -> TracedExceptionBase`, then `skip_frames`
   * should be 2.
   *
   * This class, on construction, will automatically detect any pending exceptions and "chain"
   * the current exception from it.
   */
  explicit TracedExceptionBase(const std::exception_ptr& ptr, std::size_t skip_frames);

  TracedExceptionBase(const TracedExceptionBase&)                = default;
  TracedExceptionBase& operator=(const TracedExceptionBase&)     = default;
  TracedExceptionBase(TracedExceptionBase&&) noexcept            = default;
  TracedExceptionBase& operator=(TracedExceptionBase&&) noexcept = default;
  ~TracedExceptionBase();

  /**
   * @brief Get the raw, unformatted exception message.
   *
   * @return The original exception message without any nested exceptions or traces.
   */
  [[nodiscard]] std::string_view raw_what_sv() const noexcept;

  /**
   * @brief Get the formatted exception message in c_str() form.
   *
   * @return The formatted exception containing the stack strace, suitable for derived class
   * what() methods.
   */
  [[nodiscard]] const char* traced_what() const noexcept;

  /**
   * @brief Get the formatted exception message in string_view form.
   *
   * @return The formatted exception containing the stack strace, suitable for derived class
   * what() methods.
   */
  [[nodiscard]] std::string_view traced_what_sv() const noexcept;

  [[nodiscard]] const Impl* impl() const noexcept;

 private:
  std::shared_ptr<Impl> impl_;
};

/**
 * @brief Exception wrapper which captures a traceback at the point of throwing.
 *
 * @tparam T The type of the exception to wrap, for example `std::runtime_error`.
 */
template <typename T>
class LEGATE_EXPORT TracedException : public T, public TracedExceptionBase {
 public:
  /**
   * @brief Construct a TracedException.
   *
   * @param args The arguments to construct the underlying exception object for TracedException.
   */
  template <typename... U>
  explicit TracedException(U&&... args);

  /**
   * @brief Get the formatted exception message.
   *
   * @return The formatted exception containing the stack strace.
   */
  [[nodiscard]] const char* what() const noexcept override;
};

template <typename T>
class TracedException<TracedException<T>> final {
 public:
  // This should really be static_assert(false), but we can't do that.
  static_assert(sizeof(T*) != sizeof(T*),  // NOLINT(misc-redundant-expression)
                "This overload is forbidden");
};

template <>
class TracedException<TracedExceptionBase> final {
 public:
  template <typename... T>
  explicit TracedException(T&&...);
};

template <>
class TracedException<std::bad_alloc> final {
 public:
  template <typename... T>
  explicit TracedException(T&&...);
};

}  // namespace legate::detail

#include <legate/utilities/detail/traced_exception.inl>
