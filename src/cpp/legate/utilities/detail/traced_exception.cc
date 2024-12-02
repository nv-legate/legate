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

#include <legate_defines.h>

#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/span.h>
#include <legate/version.h>

#include <legion_defines.h>

#include <array>
#include <atomic>
#include <cpptrace/basic.hpp>
#include <cpptrace/from_current.hpp>
#include <cpptrace/utils.hpp>
#include <cstdlib>
#include <exception>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <iostream>
#include <string_view>
#include <sys/utsname.h>

namespace legate::detail {

namespace {

constexpr auto BASE_ERROR_PREFIX = std::string_view{"LEGATE ERROR:"};

[[nodiscard]] bool detect_use_color() noexcept
{
  try {
    // Don't use EnvironmentVariable, because it may itself throw a TracedException. If that
    // happens, we get:
    //
    // libc++abi: __cxa_guard_acquire detected recursive initialization: do you have a
    // function-local static variable whose initialization depends on that function?
    //
    // Which is not great
    if (const auto* no_color = std::getenv("NO_COLOR"); no_color && (no_color[0] != '0')) {
      return false;
    }
    return cpptrace::isatty(cpptrace::stdout_fileno) && cpptrace::isatty(cpptrace::stderr_fileno);
  } catch (...) {
    return false;
  }
}

[[nodiscard]] std::string make_error_prefix(bool use_color)
{
  if (use_color) {
    return fmt::format(fmt::fg(fmt::color::indian_red), BASE_ERROR_PREFIX);
  }
  return std::string{BASE_ERROR_PREFIX};
}

[[nodiscard]] std::string get_system_summary()
{
  struct ::utsname res = {};

  if (::uname(&res)) {
    return "unknown system";
  }
  return fmt::format(
    "{}, {}, {}, {}, {}", res.sysname, res.release, res.nodename, res.version, res.machine);
}

void add_exception_messages(
  std::string_view PREFIX,
  Span<const std::pair<const std::type_info&, std::string_view>> prev_exns,
  std::string* ret)
{
  fmt::format_to(
    std::back_inserter(*ret), "{} {}: {}\n", PREFIX, prev_exns[0].first, prev_exns[0].second);

  if (prev_exns.size() == 1) {
    return;
  }

  fmt::format_to(
    std::back_inserter(*ret), "{} Above exception also contained nested exception(s):\n", PREFIX);
  // Start at second index
  for (std::size_t i = 1; i < prev_exns.size(); ++i) {
    fmt::format_to(std::back_inserter(*ret),
                   "{} #{} {}: {}\n",
                   PREFIX,
                   i,
                   prev_exns[i].first,
                   prev_exns[i].second);
  }
}

void add_traceback(std::string_view PREFIX,
                   bool USE_COLOR,
                   const cpptrace::stacktrace& trace,
                   std::string* ret)
{
  fmt::format_to(std::back_inserter(*ret),
                 "{} Stack trace (most recent call first, top-most exception only):\n",
                 PREFIX);
  if (trace.empty()) {
    fmt::format_to(std::back_inserter(*ret), "{} <unknown stack trace>\n", PREFIX);
    return;
  }

  const auto num_digits = [&] {
    const auto num_frames = trace.frames.size();

    if (num_frames < 10) {  // NOLINT(readability-magic-numbers)
      return 1;
    }
    if (num_frames < 100) {  // NOLINT(readability-magic-numbers)
      return 2;
    }
    if (num_frames < 1'000) {  // NOLINT(readability-magic-numbers)
      return 3;
    }
    // If we are handling tracebacks longer than 1000 function calls deep then we are in bigger
    // trouble.
    return 4;
  }();

  for (auto&& [idx, frame] : enumerate(trace)) {
    fmt::format_to(std::back_inserter(*ret),
                   "{} #{:<{}} {}\n",
                   PREFIX,
                   idx,
                   num_digits,
                   frame.to_string(USE_COLOR));
  }
}

[[nodiscard]] std::string make_error_message_impl(
  Span<const std::pair<const std::type_info&, std::string_view>> prev_exns,
  const cpptrace::stacktrace& trace)
{
  // If any of this throws an exception, we are basically screwed
  static const auto USE_COLOR      = detect_use_color();
  static const auto PREFIX         = make_error_prefix(USE_COLOR);
  constexpr auto RULER_LENGTH      = 80;
  static const auto RULER          = fmt::format("{} {:=>{}}\n", PREFIX, "", RULER_LENGTH);
  static const auto SYSTEM_SUMMARY = fmt::format("{} System: {}\n", PREFIX, get_system_summary());
  static const auto LEGATE_VERSION_SUMMARY = fmt::format("{} Legate version: {}.{}.{} ({})\n",
                                                         PREFIX,
                                                         LEGATE_VERSION_MAJOR,
                                                         LEGATE_VERSION_MINOR,
                                                         LEGATE_VERSION_PATCH,
                                                         LEGATE_GIT_HASH);
  static const auto LEGION_VERSION_SUMMARY =
    fmt::format("{} Legion version: {} ({})\n", PREFIX, LEGION_VERSION, LEGION_GIT_HASH);
  static const auto CONFIGURE_SUMMARY =
    fmt::format("{} Configure options: {}\n", PREFIX, LEGATE_CONFIGURE_OPTIONS);
  constexpr auto FIVE_KB = 5 * 1024;

  std::string ret;

  // With the system info, version info, color, and a moderately long stack trace, we easily
  // clear 5-6Kb in a single string here, so best to reserve this beast upfront. Note: the goal
  // is not performance, but to avoid fragmentation and subsequently running out of memory (if
  // that's even possible) due to constantly realloc-ing the buffer.
  ret.reserve(FIVE_KB);
  ret += RULER;
  add_exception_messages(PREFIX, prev_exns, &ret);
  ret += SYSTEM_SUMMARY;
  ret += LEGATE_VERSION_SUMMARY;
  ret += LEGION_VERSION_SUMMARY;
  ret += CONFIGURE_SUMMARY;
  add_traceback(PREFIX, USE_COLOR, trace, &ret);
  ret += RULER;
  return ret;
}

void unwrap_nested_exception(const std::exception& exn,
                             std::vector<std::pair<const std::type_info&, std::string_view>>* whats)
{
  whats->emplace_back(typeid(exn), exn.what());
  try {
    std::rethrow_if_nested(exn);
  } catch (const std::exception& nested) {
    unwrap_nested_exception(nested, whats);
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Normally we would re-throw here, but since this is called in the terminate handler, any
    // uncaught exception causes an immediate... terminatation... of the program.
  }
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
      std::cerr << exn.what_sv();
      std::abort();
    }
    catch (const std::exception& exn)
    {
      std::vector<std::pair<const std::type_info&, std::string_view>> maybe_nested;

      unwrap_nested_exception(exn, &maybe_nested);
      std::cerr << make_error_message_impl({maybe_nested.cbegin(), maybe_nested.cend()},
                                           cpptrace::from_current_exception());
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

/*static*/ std::string TracedExceptionBase::make_error_message_(const std::type_info& exn_ty,
                                                                std::string_view what,
                                                                std::size_t skip_frames)
{
  static_cast<void>(install_terminate_handler());

  const std::array<std::pair<const std::type_info&, std::string_view>, 1> tmp = {
    std::make_pair(std::cref(exn_ty), what)};

  return make_error_message_impl({tmp.begin(), tmp.end()},
                                 cpptrace::stacktrace::current(skip_frames + 1));
}

// ------------------------------------------------------------------------------------------

TracedExceptionBase::TracedExceptionBase(
  std::exception_ptr ptr,  // NOLINT(performance-unnecessary-value-param)
  const std::type_info& exn_ty,
  std::string_view what)
  // Whether or not std::exception_ptr has a move ctor is implementation defined, so we should
  // move just in case it has one
  : orig_{std::move(ptr)},  // NOLINT(performance-move-const-arg)
    what_{make_error_message_(exn_ty, what, /* skip_frames */ 1)}
{
}

}  // namespace legate::detail
