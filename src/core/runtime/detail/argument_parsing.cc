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

#include "legate_defines.h"

#include "core/mapping/detail/base_mapper.h"
#include "core/runtime/detail/config.h"
#include "core/utilities/detail/env.h"
#include "core/utilities/detail/zstring_view.h"
#include "core/utilities/macros.h"
#include "core/version.h"

#include <legion.h>

#include <argparse/argparse.hpp>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace legate::detail {

namespace {

// Simple wrapper for variables with default values
template <typename T>
class ScaledVar {
 public:
  template <typename U>
  explicit ScaledVar(U default_val, T scale = T{1});

  [[nodiscard]] T value() const;
  [[nodiscard]] constexpr T default_value() const;
  [[nodiscard]] T& ref();

 private:
  static constexpr T UNSET{std::numeric_limits<T>::max()};

  T default_value_{};
  T scale_{};
  T value_{UNSET};
};

template <typename T>
template <typename U>
ScaledVar<T>::ScaledVar(U default_val, T scale) : default_value_{default_val}, scale_{scale}
{
  static_assert(std::is_same_v<T, U>);
}

template <typename T>
T ScaledVar<T>::value() const
{
  return (value_ == UNSET ? default_value() : value_) * scale_;
}

template <typename T>
constexpr T ScaledVar<T>::default_value() const
{
  return default_value_;
}

template <typename T>
T& ScaledVar<T>::ref()
{
  return value_;
}

// ==========================================================================================

template <typename T>
class Arg {
 public:
  Arg() = delete;
  explicit Arg(std::string_view flag, T init);

  std::string_view flag{};
  T value{};
};

template <typename T>
Arg<T>::Arg(std::string_view f, T init) : flag{f}, value{std::move(init)}
{
}

// ==========================================================================================

[[nodiscard]] std::vector<std::string> split_in_args(std::string_view command)
{
  std::vector<std::string> qargs;

  // Needed to satisfy argparse, which expects an argv-like interface where argv[0] is the
  // program name.
  qargs.emplace_back("LEGATE");
  while (!command.empty()) {
    std::size_t arglen;
    auto quoted = false;

    switch (const auto c = command.front()) {
      case '\"': [[fallthrough]];
      case '\'':
        command.remove_prefix(1);
        quoted = true;
        arglen = command.find(c);
        if (arglen == std::string_view::npos) {
          throw std::invalid_argument{fmt::format("Unterminated quote: '{}'", command)};
        }
        break;
      case ' ': {
        command.remove_prefix(1);
        continue;
      }
      default:
        arglen = command.find(' ');
        if (arglen == std::string_view::npos) {
          arglen = command.size();
        }
        break;
    }

    if (auto sub = command.substr(0, arglen); !sub.empty()) {
      qargs.emplace_back(sub);
    }
    command.remove_prefix(arglen + quoted);
  }
  return qargs;
}

[[nodiscard]] std::filesystem::path normalize_log_dir(std::string log_dir)
{
  namespace fs = std::filesystem;

  auto log_path = fs::path{std::move(log_dir)};

  if (log_path.empty()) {
    log_path = fs::current_path();  // cwd
  }
  return log_path;
}

// ==========================================================================================

template <typename T>
void try_set_property(Realm::Runtime* runtime,
                      const std::string& module_name,
                      const std::string& property_name,
                      const argparse::ArgumentParser& parser,
                      const Arg<ScaledVar<T>>& var)
{
  const auto value = var.value.value();

  if (value < 0) {
    throw std::invalid_argument{fmt::format("unable to set {}", var.flag)};
  }

  auto* const config = runtime->get_module_config(module_name);

  if (nullptr == config) {
    // If the variable doesn't have a value, we don't care if the module is nonexistent
    if (!parser.is_used(var.flag)) {
      return;
    }

    throw std::runtime_error{
      fmt::format("unable to set {} (the {} module is not available)", var.flag, module_name)};
  }

  if (const auto success = config->set_property(property_name, value); !success) {
    throw std::runtime_error{fmt::format("unable to set {}", var.flag)};
  }
}

void set_core_config_properties(Realm::Runtime* rt,
                                const argparse::ArgumentParser& parser,
                                const Arg<ScaledVar<std::int32_t>>& cpus,
                                const Arg<ScaledVar<std::int32_t>>& util,
                                const Arg<ScaledVar<std::int64_t>>& sysmem,
                                const Arg<ScaledVar<std::int64_t>>& regmem)
{
  try_set_property(rt, "core", "cpu", parser, cpus);
  try_set_property(rt, "core", "util", parser, util);
  try_set_property(rt, "core", "sysmem", parser, sysmem);
  try_set_property(rt, "core", "regmem", parser, regmem);
}

void set_cuda_config_properties(Realm::Runtime* rt,
                                const argparse::ArgumentParser& parser,
                                const Arg<ScaledVar<std::int32_t>>& gpus,
                                const Arg<ScaledVar<std::int64_t>>& fbmem,
                                const Arg<ScaledVar<std::int64_t>>& zcmem)
{
  try {
    try_set_property(rt, "cuda", "gpu", parser, gpus);
    try_set_property(rt, "cuda", "fbmem", parser, fbmem);
    try_set_property(rt, "cuda", "zcmem", parser, zcmem);
  } catch (...) {
    if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
      throw;
    }
  }
  if (gpus.value.value() > 0) {
    LEGATE_NEED_CUDA.set(true);
  }
}

void set_openmp_config_properties(Realm::Runtime* rt,
                                  const argparse::ArgumentParser& parser,
                                  const Arg<ScaledVar<std::int32_t>>& omps,
                                  const Arg<ScaledVar<std::int64_t>>& ompthreads)
{
  if (omps.value.value() > 0) {
    const auto num_threads = ompthreads.value.value();

    if (num_threads <= 0) {
      throw std::invalid_argument{
        fmt::format("{} configured with zero threads: {}", omps.flag, num_threads)};
    }
    LEGATE_NEED_OPENMP.set(true);
    Config::num_omp_threads = num_threads;
  }
  try {
    try_set_property(rt, "openmp", "ocpu", parser, omps);
    try_set_property(rt, "openmp", "othr", parser, ompthreads);
  } catch (...) {
    // If we have OpenMP, but failed above, then rethrow, otherwise silently gobble the error
    if (LEGATE_DEFINED(LEGATE_USE_OPENMP)) {
      throw;
    }
  }
}

void set_legion_default_args(std::string log_dir,
                             const ScaledVar<std::int32_t>& eager_alloc_percent,
                             std::string log_levels,
                             bool profile,
                             bool spy,
                             bool freeze_on_error,
                             bool log_to_file)
{
  auto log_path = normalize_log_dir(std::move(log_dir));

  std::stringstream args_ss;

  // some values have to be passed via env var
  args_ss << "-lg:eager_alloc_percentage " << eager_alloc_percent.value() << " -lg:local 0 ";

  const auto add_logger = [&](std::string_view item) {
    if (!log_levels.empty()) {
      log_levels += ',';
    }
    log_levels += item;
  };

  LEGATE_ASSERT(Config::parsed());
  if (Config::log_mapping_decisions) {
    add_logger(fmt::format("{}=2", mapping::detail::BaseMapper::LOGGER_NAME));
  }

  if (profile) {
    args_ss << "-lg:prof 1 -lg:prof_logfile " << log_path / "legate_%.prof" << " ";
    add_logger("legion_prof=2");
  }

  if (spy) {
    args_ss << "-lg:spy ";
    add_logger("legion_spy=2");
  }

  if (freeze_on_error) {
    constexpr detail::EnvironmentVariable<std::uint32_t> LEGION_FREEZE_ON_ERROR{
      "LEGION_FREEZE_ON_ERROR"};

    LEGION_FREEZE_ON_ERROR.set(1);
    args_ss << "-ll:force_kthreads ";
  }

  if (!log_levels.empty()) {
    args_ss << "-level " << log_levels << " ";
  }

  if (log_to_file) {
    args_ss << "-logfile " << log_path / "legate_%.log" << " -errlevel 4 ";
  }

  if (const auto existing_default_args = LEGION_DEFAULT_ARGS.get();
      existing_default_args.has_value()) {
    args_ss << *existing_default_args;
  }

  LEGION_DEFAULT_ARGS.set(args_ss.str());
}

// ==========================================================================================

template <typename T>
void add_argument_base(argparse::ArgumentParser* parser,
                       std::string_view flag,
                       std::string help,
                       const T& default_val,
                       T& val)
{
  auto& parg = parser->add_argument(flag);

  parg.help(std::move(help));
  parg.default_value(default_val);
  parg.store_into(val);
  if constexpr (std::is_same_v<T, bool>) {
    parg.metavar("BOOL");
  } else {
    parg.nargs(1);
    if constexpr (std::is_integral_v<T>) {
      parg.metavar("INT");
    } else if constexpr (std::is_same_v<T, std::string>) {
      parg.metavar("STRING");
    }
  }
}

template <typename T>
[[nodiscard]] Arg<ScaledVar<T>> add_argument(argparse::ArgumentParser* parser,
                                             std::string_view flag,
                                             std::string help,
                                             ScaledVar<T> val)
{
  auto arg = Arg<ScaledVar<T>>{flag, std::move(val)};

  add_argument_base(parser, flag, std::move(help), arg.value.default_value(), arg.value.ref());
  return arg;
}

template <typename T>
[[nodiscard]] Arg<T> add_argument(argparse::ArgumentParser* parser,
                                  std::string_view flag,
                                  std::string help,
                                  T val)
{
  auto arg = Arg<T>{flag, std::move(val)};

  add_argument_base(parser, flag, std::move(help), arg.value, arg.value);
  return arg;
}

}  // namespace

void handle_legate_args(std::string_view legate_config)
{
  constexpr std::int32_t DEFAULT_CPUS                = 1;
  constexpr std::int32_t DEFAULT_GPUS                = 0;
  constexpr std::int32_t DEFAULT_OMPS                = 0;
  constexpr std::int64_t DEFAULT_OMPTHREADS          = 2;
  constexpr std::int32_t DEFAULT_UTILITY             = 1;
  constexpr std::int64_t DEFAULT_SYSMEM              = 4000;  // MB
  constexpr std::int64_t DEFAULT_NUMAMEM             = 0;     // MB
  constexpr std::int64_t DEFAULT_FBMEM               = 4000;  // MB
  constexpr std::int64_t DEFAULT_ZCMEM               = 32;    // MB
  constexpr std::int64_t DEFAULT_REGMEM              = 0;     // MB
  constexpr std::int32_t DEFAULT_EAGER_ALLOC_PERCENT = 50;
  constexpr std::int64_t MB                          = 1024 * 1024;

  auto parser = argparse::ArgumentParser{
    "LEGATE",
    std::string{LEGATE_STRINGIZE(LEGATE_VERSION_MAJOR) "." LEGATE_STRINGIZE(
      LEGATE_VERSION_MINOR) "." LEGATE_STRINGIZE(LEGATE_VERSION_PATCH)}};

  parser.add_group("Legate arguments");

  const auto cpus = add_argument(&parser,
                                 "--cpus",
                                 "number of CPU's to reserve, must be >=0",
                                 ScaledVar<std::int32_t>{DEFAULT_CPUS});
  const auto gpus = add_argument(&parser,
                                 "--gpus",
                                 "number of GPU's to reserve, must be >=0",
                                 ScaledVar<std::int32_t>{DEFAULT_GPUS});
  const auto omps = add_argument(&parser,
                                 "--omps",
                                 "number of OpenMP processors to reserve, must be >=0",
                                 ScaledVar<std::int32_t>{DEFAULT_OMPS});

  const auto ompthreads = add_argument(&parser,
                                       "--ompthreads",
                                       "number of OpenMP threads to use, must be >=0",
                                       ScaledVar<std::int64_t>{DEFAULT_OMPTHREADS});

  const auto util    = add_argument(&parser,
                                 "--utility",
                                 "number of utility processors to reserve, must be >=0",
                                 ScaledVar<std::int32_t>{DEFAULT_UTILITY});
  const auto sysmem  = add_argument(&parser,
                                   "--sysmem",
                                   "size (in megabytes) of system memory to reserve",
                                   ScaledVar<std::int64_t>{DEFAULT_SYSMEM, MB});
  const auto numamem = add_argument(&parser,
                                    "--numamem",
                                    "size (in megabytes) of NUMA memory to reserve",
                                    ScaledVar<std::int64_t>{DEFAULT_NUMAMEM, MB});
  const auto fbmem =
    add_argument(&parser,
                 "--fbmem",
                 "size (in megabytes) of GPU (or \"frame buffer\") memory to reservef",
                 ScaledVar<std::int64_t>{DEFAULT_FBMEM, MB});
  const auto zcmem  = add_argument(&parser,
                                  "--zcmem",
                                  "size (in megabytes) of zero-copy GPU memory to reserve",
                                  ScaledVar<std::int64_t>{DEFAULT_ZCMEM, MB});
  const auto regmem = add_argument(&parser,
                                   "--regmem",
                                   "size (in megamytes) in regmem to reserve",
                                   ScaledVar<std::int64_t>{DEFAULT_REGMEM, MB});

  const auto eager_alloc_percent =
    add_argument(&parser,
                 "--eager-alloc-percentage",
                 "percentage of eager allocation",
                 ScaledVar<std::int32_t>{DEFAULT_EAGER_ALLOC_PERCENT});

  const auto profile =
    add_argument(&parser, "--profile", "whether to enable Legion runtime profiling", false);
  const auto spy  = add_argument(&parser, "--spy", "whether to enable Legion spy", false);
  auto log_levels = add_argument(
    &parser,
    "--logging",
    "comma separated list of loggers to enable and their level, e.g. legate=3,foo=0,bar=5",
    std::string{});
  auto log_dir = add_argument(&parser, "--logdir", "directory to emit logfiles to", std::string{});
  const auto log_to_file =
    add_argument(&parser, "--log-to-file", "wether to save logs to file", false);
  const auto freeze_on_error = add_argument(
    &parser, "--freeze-on-error", "whether to pause the program on first error", false);

  {
    auto&& args = split_in_args(legate_config);

    try {
      parser.parse_args(args);
    } catch (const std::exception& exn) {
      std::cerr << "== LEGATE ERROR:\n";
      std::cerr << "== LEGATE ERROR: " << exn.what() << "\n";
      std::cerr << "== LEGATE ERROR:\n";
      std::cerr << parser;
      std::exit(EXIT_FAILURE);
    }
  }

  auto rt = Realm::Runtime::get_runtime();

  // ensure core module
  if (!rt.get_module_config("core")) {
    throw std::runtime_error{"core module config is missing"};
  }

  // ensure sensible utility
  if (const auto nutil = util.value.value(); nutil < 1) {
    throw std::invalid_argument{fmt::format("{} must be at least 1 (have {})", util.flag, nutil)};
  }

  // Set core configuration properties
  set_core_config_properties(&rt, parser, cpus, util, sysmem, regmem);

  // Set CUDA configuration properties
  set_cuda_config_properties(&rt, parser, gpus, fbmem, zcmem);

  // Set OpenMP configuration properties
  set_openmp_config_properties(&rt, parser, omps, ompthreads);

  // Set NUMA configuration properties
  try_set_property(&rt, "numa", "numamem", parser, numamem);

  set_legion_default_args(std::move(log_dir.value),
                          eager_alloc_percent.value,
                          std::move(log_levels.value),
                          profile.value,
                          spy.value,
                          freeze_on_error.value,
                          log_to_file.value);
}

}  // namespace legate::detail
