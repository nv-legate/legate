/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/config_legion.h>

#include <legate_defines.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/flags/logging.h>
#include <legate/runtime/detail/argument_parsing/parse.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

#include <fmt/std.h>

#include <cstdint>
#include <iterator>
#include <string>

namespace legate::detail {

std::string compose_legion_default_args(const ParsedArgs& parsed)
{
  std::string ret = "-lg:local 0 ";

  // Turn point-wise analysis ON in Legion by default
  ret += "-lg:enable_pointwise_analysis ";

  // Turn off automatic trace detection from running in the background, to avoid any interference
  // from the added analysis overhead. Legion should already be very conservative with allocating
  // resources to this task, but disabling it just to be safe. Notably this is also disabled in the
  // Legate base mapper.
  ret += "-lg:no_auto_tracing ";

  // If these are negative, then we forgot to configure them
  LEGATE_CHECK(parsed.omps.value() >= 0);
  LEGATE_CHECK(parsed.numamem.value().unscaled_value() >= 0);
  if (parsed.omps.value() >= 1 && parsed.numamem.value().scaled_value() == 0) {
    // Realm will try to allocate OpenMP groups in a NUMA-aligned way, even if NUMA detection
    // failed (in which case the auto-configuration system set --numamem 0), resulting in a warning.
    // Just tell it to not bother, so we suppress the warning.
    // Technically speaking it might be useful to enable NUMA-aligned OpenMP group instantiation
    // in cases where NUMA is available, but we're explicitly requesting no NUMA-aligned memory,
    // i.e. the user set --numamem 0.
    ret += "-ll:onuma 0 ";
  }

  if (parsed.profile.value()) {
    fmt::format_to(std::back_inserter(ret),
                   "-lg:prof 1 "
                   "-lg:prof_logfile \"{}_%.prof\" ",
                   parsed.log_dir.value() / parsed.profile_name.value());
  }

  if (auto&& log_levels = parsed.log_levels.value(); !log_levels.empty()) {
    fmt::format_to(std::back_inserter(ret), "-level {} ", convert_log_levels(log_levels));
  }

  if (parsed.log_to_file.value()) {
    fmt::format_to(std::back_inserter(ret),
                   "-logfile \"{}\" "
                   "-errlevel 4 ",
                   parsed.log_dir.value() / "legate_%.log");
  }

  if (LEGATE_DEFINED(LEGATE_HAS_ASAN) || parsed.freeze_on_error.value()) {
    // TODO (wonchanl, jfaibussowit) Sanitizers can raise false alarms if the code does
    // user-level threading, so we turn it off for sanitizer-enabled tests
    ret += "-ll:force_kthreads ";
  }

  if (const auto existing_default_args = LEGION_DEFAULT_ARGS.get();
      existing_default_args.has_value()) {
    ret += ' ';
    ret += *existing_default_args;
  }
  return ret;
}

void configure_legion(const ParsedArgs& parsed)
{
  const auto legion_args = compose_legion_default_args(parsed);

  LEGION_DEFAULT_ARGS.set(legion_args);
  if (parsed.freeze_on_error.value()) {
    constexpr EnvironmentVariable<std::uint32_t> LEGION_FREEZE_ON_ERROR{"LEGION_FREEZE_ON_ERROR"};

    LEGION_FREEZE_ON_ERROR.set(/*value=*/1);
  }
}

}  // namespace legate::detail
