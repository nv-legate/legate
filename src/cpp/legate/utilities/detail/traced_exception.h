/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <exception>
#include <new>
#include <string>
#include <string_view>
#include <typeinfo>

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
class TracedExceptionBase {
 public:
  /**
   * @brief Construct a TracedExceptionBase.
   *
   * @param ptr An exception pointer to the original exception object.
   * @param exn_ty The type info of the original exception object.
   * @param args The arguments to construct the underlying exception object for TracedException.
   */
  TracedExceptionBase(std::exception_ptr ptr, const std::type_info& exn_ty, std::string_view what);

  /**
   * @brief Get the formatted exception message in c_str() form.
   *
   * @return The formatted exception containing the stack strace, suitable for derived class
   * what() methods.
   */
  [[nodiscard]] const char* what() const noexcept;

  /**
   * @brief Get the formatted exception message.
   *
   * @return The formatted exception containing the stack strace.
   */
  [[nodiscard]] std::string_view what_sv() const noexcept;

  /**
   * @brief Get the original exception being traced.
   *
   * @return The original exception.
   */
  [[nodiscard]] std::exception_ptr original_exception() const noexcept;

 private:
  [[nodiscard]] static std::string make_error_message_(const std::type_info& exn_ty,
                                                       std::string_view what,
                                                       std::size_t skip_frames);

  std::exception_ptr orig_{};
  std::string what_{};
};

/**
 * @brief Exception wrapper which captures a traceback at the point of throwing.
 *
 * @tparam T The type of the exception to wrap, for example `std::runtime_error`.
 */
template <typename T>
class TracedException : public T, public TracedExceptionBase {
 public:
  /**
   * @brief Construct a TracedException
   *
   * @param args The arguments to construct the underlying exception object for TracedException.
   */
  template <typename... U>
  constexpr explicit TracedException(U&&... args);

  /**
   * @brief Get the formatted exception message.
   *
   * @return The formatted exception containing the stack strace.
   */
  [[nodiscard]] const char* what() const noexcept override;

 private:
  std::string what_{};
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
