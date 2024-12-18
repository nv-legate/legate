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

#include <legate/utilities/detail/error.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/span.h>

#include <array>
#include <atomic>
#include <cpptrace/basic.hpp>
#include <cpptrace/from_current.hpp>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fmt/format.h>
#include <fmt/std.h>
#include <iostream>
#include <string_view>
#include <vector>

namespace legate::detail {

namespace {

void unwrap_nested_exception(const std::exception& exn, std::vector<std::string>* whats)
{
  switch (const auto num_nested = whats->size()) {
    case 0: whats->emplace_back(fmt::format("{}: {}", typeid(exn), exn.what())); break;
    case 1:
      whats->emplace_back("Above exception also contained nested exception(s):");
      [[fallthrough]];
    default:
      whats->emplace_back(fmt::format("#{} {}: {}", num_nested, typeid(exn), exn.what()));
      break;
  }

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
      std::vector<std::string> maybe_nested;

      unwrap_nested_exception(exn, &maybe_nested);
      std::cerr << make_error_message({maybe_nested.cbegin(), maybe_nested.cend()},
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

  const std::array<std::string, 1> tmp = {fmt::format("{}: {}", exn_ty, what)};

  return make_error_message({tmp.cbegin(), tmp.cend()},
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
