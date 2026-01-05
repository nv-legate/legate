/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/traced_exception.h>

#include <legate/utilities/abort.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/error.h>
#include <legate/utilities/span.h>

#include <fmt/format.h>
#include <fmt/std.h>

#include <cpptrace/basic.hpp>
#include <cpptrace/from_current.hpp>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace legate::detail {

namespace {

// TODO(jfaibussowit):
// Need to unify this with the unwrapping code from TracedExceptionBase. Currently this
// function only properly handles pure std exception nesting, and the other only handles pure
// TracedException nesting. But in theory someone could std::throw_with_nested(TracedException)
// and then this kind of falls apart.
void unwrap_nested_exception(const std::exception& exn,
                             std::size_t depth,
                             std::vector<ErrorDescription>* errs)
{
  CPPTRACE_TRY { std::rethrow_if_nested(exn); }
  CPPTRACE_CATCH (const std::exception& nested) { unwrap_nested_exception(nested, depth + 1, errs); }
  catch (...)  // NOLINT(bugprone-empty-catch)
  {
    // Normally we would re-throw here, but since this is called in the terminate handler, any
    // uncaught exception causes an immediate... terminatation... of the program.
  }
  errs->reserve(depth);
  // Order is important here. make_error_message() expects to iterate "bottom up", so start
  // from the most nested and end with the top-most exception. So we need to recurse *first*
  // before adding our own stuff to it.
  errs->emplace_back(fmt::format("{}: {}", typeid(exn), exn.what()),
                     cpptrace::from_current_exception());
}

[[nodiscard]] std::terminate_handler get_terminate_handler() noexcept
{
  // Must be static, otherwise the handler cannot "capture" it.
  static const auto prev_handler = std::get_terminate();
  constexpr auto handler         = [] {
    CPPTRACE_TRY
    {
      if (const auto eptr = std::current_exception()) {
        std::rethrow_exception(eptr);
      }
      // fall-through
    }
    CPPTRACE_CATCH(const TracedExceptionBase& exn)
    {
      std::cerr << exn.traced_what_sv();
      std::abort();
    }
    catch (const std::exception& exn)
    {
      std::vector<ErrorDescription> errs;

      unwrap_nested_exception(exn, /* depth */ 1, &errs);
      std::cerr << make_error_message(errs);
      std::abort();
    }
    catch (...)
    {
      try {
        cpptrace::from_current_exception().print(std::cerr);
        std::abort();
      } catch (...) {  //  NOLINT(bugprone-empty-catch)
      }
      // Don't fall through because the previous handler might expect to handle in-flight
      // exceptions, so we need to pretend like the original exception is still in flight. If
      // we exit this catch() clause, then we will have "handled" the original exception, and
      // current_exception() will return NULL for the other handler.
      prev_handler();
    }
    prev_handler();
  };

  return handler;
}

}  // namespace

bool install_terminate_handler() noexcept
{
  // C++11 guarantees that static variables are initialized in a thread-safe manner, so by
  // capturing the return value in a static variable, we ensure that this entire function is
  // done not only once (and only once), but that it is thread-safe.
  static const auto _ = std::set_terminate(get_terminate_handler());
  static_cast<void>(_);
  static std::atomic_flag installed = ATOMIC_FLAG_INIT;

  return !installed.test_and_set(std::memory_order_relaxed);
}

// ==========================================================================================

class TracedExceptionBase::Impl : public std::nested_exception {
 public:
  Impl(const std::exception_ptr& exn, cpptrace::raw_trace trace);

  [[nodiscard]] const std::exception_ptr& exception() const noexcept;
  [[nodiscard]] const cpptrace::raw_trace& trace() const noexcept;

  [[nodiscard]] std::string_view raw_what_sv() const noexcept;
  [[nodiscard]] const char* traced_what() const noexcept;
  [[nodiscard]] std::string_view traced_what_sv() const noexcept;

 private:
  void unwrap_nested_(std::size_t depth, std::vector<ErrorDescription>* errs) const;
  void ensure_traced_what_() const;

  std::exception_ptr exn_{};
  cpptrace::raw_trace trace_{};
  mutable std::optional<std::string> what_{};
};

void TracedExceptionBase::Impl::unwrap_nested_(std::size_t depth,
                                               std::vector<ErrorDescription>* errs) const
{
  if (nested_ptr()) {
    try {
      rethrow_nested();
    } catch (const TracedExceptionBase& traced) {
      traced.impl()->unwrap_nested_(depth + 1, errs);
    } catch (const std::exception& e) {
      // We have reached the bottom of the stack.
      errs->reserve(depth + 1);
      errs->emplace_back(fmt::format("{}: {}", typeid(e), e.what()));
    } catch (...) {
      LEGATE_ABORT("Nested exception not derived from std::exception");
    }
  } else {
    // We have also reached the bottom of the stack.
    errs->reserve(depth);
  }

  // Order is important here. make_error_message() expects to iterate "bottom up", so start
  // from the most nested and end with the top-most exception. So we need to recurse *first*
  // before adding our own stuff to it.
  try {
    std::rethrow_exception(exception());
  } catch (const std::exception& e) {
    errs->emplace_back(fmt::format("{}: {}", typeid(e), e.what()), trace().resolve());
  } catch (...) {
    LEGATE_ABORT("Original exception not derived from std::exception");
  }
}

void TracedExceptionBase::Impl::ensure_traced_what_() const
{
  if (!what_.has_value()) {
    std::vector<ErrorDescription> errs;

    unwrap_nested_(/* depth */ 1, &errs);
    what_ = make_error_message(errs);
  }
}

// ------------------------------------------------------------------------------------------

TracedExceptionBase::Impl::Impl(
  // exception_ptr literally does not have a move-ctor
  const std::exception_ptr& exn,  // NOLINT(modernize-pass-by-value)
  cpptrace::raw_trace trace)
  // clang-tidy complains that we aren't throwing the exception pointer here:
  //
  // src/cpp/legate/utilities/detail/traced_exception.cc:188:5: error: suspicious exception
  // object created but not thrown; did you mean 'throw exception_ptr'?
  // [bugprone-throw-keyword-missing,-warnings-as-errors]
  //     |
  // 188 |   : exn_{std::move(exn)},
  //     |     ^
  //
  // Which is clearly delusional...
  : exn_{exn},  // NOLINT(bugprone-throw-keyword-missing)
    trace_{std::move(trace)}
{
  LEGATE_CHECK(exception());
}

const std::exception_ptr& TracedExceptionBase::Impl::exception() const noexcept { return exn_; }

const cpptrace::raw_trace& TracedExceptionBase::Impl::trace() const noexcept { return trace_; }

std::string_view TracedExceptionBase::Impl::raw_what_sv() const noexcept
{
  try {
    std::rethrow_exception(exception());
  } catch (const TracedExceptionBase& e) {
    LEGATE_ABORT("Exception must not be a traced exception");
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    LEGATE_ABORT("Original exception not derived from std::exception");
  }
}

const char* TracedExceptionBase::Impl::traced_what() const noexcept
{
  ensure_traced_what_();
  return what_->c_str();  // NOLINT(bugprone-unchecked-optional-access)
}

std::string_view TracedExceptionBase::Impl::traced_what_sv() const noexcept
{
  ensure_traced_what_();
  return *what_;  // NOLINT(bugprone-unchecked-optional-access)
}

// ------------------------------------------------------------------------------------------

TracedExceptionBase::TracedExceptionBase(const std::exception_ptr& ptr, std::size_t skip_frames)
  : impl_{std::make_shared<Impl>(ptr, cpptrace::raw_trace::current(skip_frames + 1))}
{
}

std::string_view TracedExceptionBase::raw_what_sv() const noexcept { return impl()->raw_what_sv(); }

const char* TracedExceptionBase::traced_what() const noexcept { return impl()->traced_what(); }

std::string_view TracedExceptionBase::traced_what_sv() const noexcept
{
  return impl()->traced_what_sv();
}

TracedExceptionBase::~TracedExceptionBase() = default;

}  // namespace legate::detail
