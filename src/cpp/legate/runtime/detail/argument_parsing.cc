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

#include "legate/runtime/detail/argument_parsing.h"

#include "legate_defines.h"

#include "legate/mapping/detail/base_mapper.h"
#include "legate/runtime/detail/config.h"
#include "legate/utilities/detail/env.h"
#include "legate/utilities/detail/zstring_view.h"
#include "legate/utilities/macros.h"
#include "legate/version.h"
#include <legate/utilities/detail/traced_exception.h>

#include <legion.h>

#include <argparse/argparse.hpp>
#include <cmath>
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

constexpr std::int64_t MB = 1 << 20;

// Simple wrapper for variables with default values
template <typename T>
class ScaledVar {
 public:
  template <typename U>
  explicit ScaledVar(U default_val, T scale = T{1});

  using value_type = T;

  [[nodiscard]] T raw_value() const;
  [[nodiscard]] T value() const;
  [[nodiscard]] constexpr T default_value() const;
  [[nodiscard]] T& ref();
  void set(T value);

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
T ScaledVar<T>::raw_value() const
{
  return (value_ == UNSET ? default_value() : value_);
}

template <typename T>
T ScaledVar<T>::value() const
{
  return raw_value() * scale_;
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

template <typename T>
void ScaledVar<T>::set(T value)
{
  value_ = std::move(value);
}

// ==========================================================================================

template <typename T>
class Arg {
 public:
  Arg() = delete;
  explicit Arg(std::string_view flag, T init);

  using value_type = T;

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
          throw TracedException<std::invalid_argument>{
            fmt::format("Unterminated quote: '{}'", command)};
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
    throw TracedException<std::invalid_argument>{
      fmt::format("unable to set {} to {}", var.flag, value)};
  }

  auto* const config = runtime->get_module_config(module_name);

  if (nullptr == config) {
    // If the variable doesn't have a value, we don't care if the module is nonexistent
    if (!parser.is_used(var.flag)) {
      return;
    }

    throw TracedException<std::runtime_error>{
      fmt::format("unable to set {} (the {} module is not available)", var.flag, module_name)};
  }

  if (!config->set_property(property_name, value)) {
    throw TracedException<std::runtime_error>{fmt::format("unable to set {}", var.flag)};
  }
}

// ==========================================================================================

constexpr std::int64_t MINIMAL_MEM = 256;  // MB
constexpr double SYSMEM_FRACTION   = 0.8;

void autoconfigure_gpus(const Realm::ModuleConfig* cuda, ScaledVar<std::int32_t>* gpus)
{
  if (gpus->value() >= 0) {
    return;
  }

  std::decay_t<decltype(*gpus)>::value_type auto_gpus = 0;

  if (Config::auto_config && cuda != nullptr) {
    // use all available GPUs
    if (!cuda->get_resource("gpu", auto_gpus)) {
      throw TracedException<std::runtime_error>{
        "Auto-configuration failed: CUDA Realm module could not determine the number of GPUs"};
    }
  }  // otherwise don't allocate any GPUs
  gpus->set(auto_gpus);
}

void autoconfigure_fbmem(const Realm::ModuleConfig* cuda,
                         std::int32_t gpus,
                         ScaledVar<std::int64_t>* fbmem)
{
  if (fbmem->value() >= 0) {
    return;
  }

  if (gpus <= 0) {
    fbmem->set(0);
    return;
  }

  if (Config::auto_config) {
    constexpr double FBMEM_FRACTION = 0.95;
    std::size_t res_fbmem;

    if (!cuda->get_resource("fbmem", res_fbmem)) {
      throw TracedException<std::runtime_error>{
        "Auto-configuration failed: CUDA Realm module could not determine the available GPU "
        "memory"};
    }

    using T = std::decay_t<decltype(*fbmem)>::value_type;

    const auto auto_fbmem =
      static_cast<T>(std::floor(FBMEM_FRACTION * static_cast<double>(res_fbmem) / MB));

    fbmem->set(auto_fbmem);
  } else {
    fbmem->set(MINIMAL_MEM);
  }
}

void autoconfigure_omps(const Realm::ModuleConfig* openmp,
                        const std::vector<std::size_t>& numa_mems,
                        std::int32_t gpus,
                        ScaledVar<std::int32_t>* omps)
{
  if (omps->value() >= 0) {
    return;
  }

  using T = std::decay_t<decltype(*omps)>::value_type;

  const auto auto_omps = [&]() -> T {
    if (!Config::auto_config || !openmp) {
      return 0;  // don't allocate any OpenMP groups
    }
    if (gpus > 0) {
      // match the number of GPUs, to ensure host offloading does not repartition
      return gpus;
    }
    // create one OpenMP group per NUMA node (or a single group, if no NUMA info is available)
    return std::max(static_cast<T>(numa_mems.size()), T{1});
  }();

  omps->set(auto_omps);
}

void autoconfigure_numamem(const std::vector<std::size_t>& numa_mems,
                           std::int32_t omps,
                           ScaledVar<std::int64_t>* numamem)
{
  if (numamem->value() >= 0) {
    return;
  }

  if (omps <= 0 || numa_mems.empty()) {
    numamem->set(0);
    return;
  }

  if (!Config::auto_config) {
    numamem->set(MINIMAL_MEM);
    return;
  }

  // TODO(mpapadakis): Assuming that all NUMA domains have the same size
  const auto numa_mem_size = numa_mems.front();
  const auto num_numa_mems = numa_mems.size();
  const auto omps_per_numa = (omps + num_numa_mems - 1) / num_numa_mems;
  using T                  = std::decay_t<decltype(*numamem)>::value_type;
  const auto auto_numamem =
    static_cast<T>(std::floor(SYSMEM_FRACTION * static_cast<double>(numa_mem_size) / MB /
                              static_cast<double>(omps_per_numa)));

  numamem->set(auto_numamem);
}

void autoconfigure_cpus(const Realm::ModuleConfig* core,
                        std::int32_t omps,
                        std::int32_t util,
                        std::int32_t gpus,
                        ScaledVar<std::int32_t>* cpus)
{
  if (cpus->value() >= 0) {
    return;
  }

  if (!Config::auto_config || (omps > 0)) {
    // leave one core available for profiling meta-tasks, and other random uses
    cpus->set(1);
    return;
  }

  if (gpus > 0) {
    // match the number of GPUs, to ensure host offloading does not repartition
    cpus->set(gpus);
    return;
  }

  // use all unallocated cores
  int res_num_cpus{};

  if (!core->get_resource("cpu", res_num_cpus)) {
    throw TracedException<std::runtime_error>{
      "Auto-configuration failed: Core Realm module could not determine the number of CPU "
      "cores"};
  }
  if (res_num_cpus == 0) {
    throw TracedException<std::runtime_error>{
      "Auto-configuration failed: Core Realm module detected 0 CPU cores while configuring "
      "CPUs"};
  }

  using T = std::decay_t<decltype(*cpus)>::value_type;

  const T auto_cpus = res_num_cpus - util - gpus;

  if (auto_cpus <= 0) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Auto-configuration failed: No CPU cores left to allocate to CPU processors. "
                  "Have {}, but need {} for utility processors, and {} for GPU processors.",
                  res_num_cpus,
                  util,
                  gpus)};
  }
  cpus->set(auto_cpus);
}

void autoconfigure_sysmem(const Realm::ModuleConfig* core,
                          std::int64_t numamem,
                          ScaledVar<std::int64_t>* sysmem)
{
  if (sysmem->value() >= 0) {
    return;
  }

  if (!Config::auto_config || (numamem > 0)) {
    // don't allocate much memory to --sysmem; leave most to be used for --numamem
    sysmem->set(MINIMAL_MEM);
    return;
  }

  using T = std::decay_t<decltype(*sysmem)>::value_type;

  std::size_t res_sysmem_size{};

  if (!core->get_resource("sysmem", res_sysmem_size)) {
    throw TracedException<std::runtime_error>{
      "Auto-configuration failed: Core Realm module could not determine the available system "
      "memory"};
  }

  const auto auto_sysmem =
    static_cast<T>(std::floor(SYSMEM_FRACTION * static_cast<double>(res_sysmem_size) / MB));

  sysmem->set(auto_sysmem);
}

void autoconfigure_ompthreads(const Realm::ModuleConfig* core,
                              std::int32_t util,
                              std::int32_t cpus,
                              std::int32_t gpus,
                              std::int32_t omps,
                              ScaledVar<std::int32_t>* ompthreads)
{
  if (ompthreads->value() >= 0) {
    return;
  }

  if (omps <= 0) {
    ompthreads->set(0);
    return;
  }

  if (!Config::auto_config) {
    ompthreads->set(1);
    return;
  }

  using T = std::decay_t<decltype(*ompthreads)>::value_type;

  int res_num_cpus{};

  if (!core->get_resource("cpu", res_num_cpus)) {
    throw TracedException<std::runtime_error>{
      "Auto-configuration failed: Core Realm module could not determine the number of CPU "
      "cores"};
  }
  if (res_num_cpus == 0) {
    throw TracedException<std::runtime_error>{
      "Auto-configuration failed: Core Realm module detected 0 CPU cores while configuring "
      "number of OpenMP threads"};
  }

  const auto auto_ompthreads =
    static_cast<T>(std::floor((res_num_cpus - cpus - util - gpus) / omps));

  if (auto_ompthreads <= 0) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Auto-configuration failed: Not enough CPU cores to split across {} OpenMP "
                  "processors. Have {}, but need {} for CPU processors, {} for utility "
                  "processors, and {} for GPU processors.",
                  omps,
                  res_num_cpus,
                  cpus,
                  util,
                  gpus)};
  }
  ompthreads->set(auto_ompthreads);
}

void autoconfigure(Realm::Runtime* rt,
                   Arg<ScaledVar<std::int32_t>>* util,
                   Arg<ScaledVar<std::int32_t>>* cpus,
                   Arg<ScaledVar<std::int32_t>>* gpus,
                   Arg<ScaledVar<std::int32_t>>* omps,
                   Arg<ScaledVar<std::int32_t>>* ompthreads,
                   Arg<ScaledVar<std::int64_t>>* sysmem,
                   Arg<ScaledVar<std::int64_t>>* fbmem,
                   Arg<ScaledVar<std::int64_t>>* numamem)
{
  const auto* core = rt->get_module_config("core");

  // ensure core module
  if (core == nullptr) {
    throw TracedException<std::runtime_error>{"core module config is missing"};
  }

  const auto* cuda   = rt->get_module_config("cuda");
  const auto* openmp = rt->get_module_config("openmp");
  const auto* numa   = rt->get_module_config("numa");
  std::vector<std::size_t> numa_mems{};
  if (numa && !numa->get_resource("numa_mems", numa_mems)) {
    numa_mems.clear();
  }

  // auto-configure --gpus
  autoconfigure_gpus(cuda, &gpus->value);

  // auto-configure --fbmem
  autoconfigure_fbmem(cuda, gpus->value.value(), &fbmem->value);

  // auto-configure --omps
  autoconfigure_omps(openmp, numa_mems, gpus->value.value(), &omps->value);

  // auto-configure --numamem
  autoconfigure_numamem(numa_mems, omps->value.value(), &numamem->value);

  // auto-configure --cpus
  autoconfigure_cpus(
    core, omps->value.value(), util->value.value(), gpus->value.value(), &cpus->value);

  // auto-configure --sysmem
  autoconfigure_sysmem(core, numamem->value.value(), &sysmem->value);

  // auto-configure --ompthreads
  autoconfigure_ompthreads(core,
                           util->value.value(),
                           cpus->value.value(),
                           gpus->value.value(),
                           omps->value.value(),
                           &ompthreads->value);
}

// ==========================================================================================

void set_core_config_properties(Realm::Runtime* rt,
                                const argparse::ArgumentParser& parser,
                                const Arg<ScaledVar<std::int32_t>>& cpus,
                                const Arg<ScaledVar<std::int32_t>>& util,
                                const Arg<ScaledVar<std::int64_t>>& sysmem,
                                const Arg<ScaledVar<std::int64_t>>& regmem)
{
  constexpr std::size_t SYSMEM_LIMIT_FOR_IPC_REG = 1024 * MB;

  try_set_property(rt, "core", "cpu", parser, cpus);
  try_set_property(rt, "core", "util", parser, util);
  try_set_property(rt, "core", "sysmem", parser, sysmem);
  try_set_property(rt, "core", "regmem", parser, regmem);

  // Don't register sysmem for intra-node IPC if it's above a certain size, as it can take forever.
  auto* const config = rt->get_module_config("core");
  LEGATE_CHECK(config != nullptr);
  static_cast<void>(config->set_property("sysmem_ipc_limit", SYSMEM_LIMIT_FOR_IPC_REG));
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
    // If we have CUDA, but failed above, then rethrow, otherwise silently gobble the error
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
                                  const Arg<ScaledVar<std::int32_t>>& ompthreads,
                                  const Arg<ScaledVar<std::int64_t>>& numamem)
{
  if (omps.value.value() > 0) {
    const auto num_threads = ompthreads.value.value();

    if (num_threads <= 0) {
      throw TracedException<std::invalid_argument>{
        fmt::format("{} configured with zero threads: {}", omps.flag, num_threads)};
    }
    LEGATE_NEED_OPENMP.set(true);
    Config::num_omp_threads = num_threads;
  }
  try {
    try_set_property(rt, "openmp", "ocpu", parser, omps);
    try_set_property(rt, "openmp", "othr", parser, ompthreads);
    try_set_property(rt, "numa", "numamem", parser, numamem);
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

  if (spy) {
    if (!log_to_file && !log_levels.empty()) {
      // Spy output is dumped to the same place as other logging, so we must redirect all logging to
      // a file, even if the user didn't ask for it.
      // NOLINTNEXTLINE(performance-avoid-endl) endl is deliberate, we want a newline and flush
      std::cout << "Logging output is being redirected to a file in --logdir" << std::endl;
    }
    args_ss << "-lg:spy ";
    add_logger("legion_spy=2");
    log_to_file = true;
  }

  // Do this after the --spy w/o --logdir check above, as the logging level legion_prof=2 doesn't
  // actually print anything to the logs, so don't consider that a conflict.
  if (profile) {
    args_ss << "-lg:prof 1 -lg:prof_logfile " << log_path / "legate_%.prof" << " ";
    add_logger("legion_prof=2");
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

  if (LEGATE_DEFINED(LEGATE_HAS_ASAN)) {
    // TODO (wonchanl, jfaibussowit) Sanitizers can raise false alarms if the code does
    // user-level threading, so we turn it off for sanitizer-enabled tests
    args_ss << "-ll:force_kthreads ";
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

void handle_legate_args()
{
  // values with -1 defaults will be auto-configured via the Realm API
  constexpr std::int32_t DEFAULT_CPUS                = -1;
  constexpr std::int32_t DEFAULT_GPUS                = -1;
  constexpr std::int32_t DEFAULT_OMPS                = -1;
  constexpr std::int32_t DEFAULT_OMPTHREADS          = -1;
  constexpr std::int32_t DEFAULT_UTILITY             = 2;
  constexpr std::int64_t DEFAULT_SYSMEM              = -1;
  constexpr std::int64_t DEFAULT_NUMAMEM             = -1;
  constexpr std::int64_t DEFAULT_FBMEM               = -1;
  constexpr std::int64_t DEFAULT_ZCMEM               = 128;  // MB
  constexpr std::int64_t DEFAULT_REGMEM              = 0;    // MB
  constexpr std::int32_t DEFAULT_EAGER_ALLOC_PERCENT = 50;

  auto parser = argparse::ArgumentParser{
    "LEGATE_CONFIG can contain:",
    std::string{LEGATE_STRINGIZE(LEGATE_VERSION_MAJOR) "." LEGATE_STRINGIZE(
      LEGATE_VERSION_MINOR) "." LEGATE_STRINGIZE(LEGATE_VERSION_PATCH)}};

  auto cpus = add_argument(&parser,
                           "--cpus",
                           "Number of standalone CPU cores to reserve, must be >=0",
                           ScaledVar<std::int32_t>{DEFAULT_CPUS});
  auto gpus = add_argument(&parser,
                           "--gpus",
                           "Number of GPUs to reserve, must be >=0",
                           ScaledVar<std::int32_t>{DEFAULT_GPUS});
  auto omps = add_argument(&parser,
                           "--omps",
                           "Number of OpenMP groups to use, must be >=0",
                           ScaledVar<std::int32_t>{DEFAULT_OMPS});

  auto ompthreads =
    add_argument(&parser,
                 "--ompthreads",
                 "Number of threads / reserved CPU cores per OpenMP group, must be >=0",
                 ScaledVar<std::int32_t>{DEFAULT_OMPTHREADS});

  auto util   = add_argument(&parser,
                           "--utility",
                           "Number of threads to use for runtime meta-work, must be >=0",
                           ScaledVar<std::int32_t>{DEFAULT_UTILITY});
  auto sysmem = add_argument(&parser,
                             "--sysmem",
                             "Size (in MiB) of DRAM memory to reserve per rank",
                             ScaledVar<std::int64_t>{DEFAULT_SYSMEM, MB});
  auto numamem =
    add_argument(&parser,
                 "--numamem",
                 "Size (in MiB) of NUMA-specific DRAM memory to reserve per NUMA domain",
                 ScaledVar<std::int64_t>{DEFAULT_NUMAMEM, MB});
  auto fbmem = add_argument(&parser,
                            "--fbmem",
                            "Size (in MiB) of GPU (or \"framebuffer\") memory to reserve per GPU",
                            ScaledVar<std::int64_t>{DEFAULT_FBMEM, MB});
  auto zcmem = add_argument(
    &parser,
    "--zcmem",
    "Size (in MiB) of GPU-registered (or \"zero-copy\") DRAM memory to reserve per GPU",
    ScaledVar<std::int64_t>{DEFAULT_ZCMEM, MB});
  auto regmem = add_argument(&parser,
                             "--regmem",
                             "Size (in MiB) of NIC-registered DRAM memory to reserve",
                             ScaledVar<std::int64_t>{DEFAULT_REGMEM, MB});

  const auto eager_alloc_percent =
    add_argument(&parser,
                 "--eager-alloc-percentage",
                 "Percentage of reserved memory to allocate for eager allocations",
                 ScaledVar<std::int32_t>{DEFAULT_EAGER_ALLOC_PERCENT});

  const auto profile =
    add_argument(&parser, "--profile", "Whether to collect profiling logs", false);
  const auto spy =
    add_argument(&parser, "--spy", "Whether to collect dataflow & task graph logs", false);
  auto log_levels = add_argument(
    &parser,
    "--logging",
    "Comma separated list of loggers to enable and their level, e.g. legate=3,foo=0,bar=5",
    std::string{});
  auto log_dir           = add_argument(&parser,
                              "--logdir",
                              "Directory to emit logfiles to, defaults to current directory",
                              std::string{});
  const auto log_to_file = add_argument(
    &parser, "--log-to-file", "Redirect logging output to a file inside --logdir", false);
  const auto freeze_on_error = add_argument(
    &parser,
    "--freeze-on-error",
    "If the program crashes, freeze execution right before exit so a debugger can be attached",
    false);

  auto legate_config_env = LEGATE_CONFIG.get();
  auto&& args            = split_in_args(legate_config_env.value_or(""));

  try {
    parser.parse_args(args);
  } catch (const std::exception& exn) {
    std::cerr << "== LEGATE ERROR:\n";
    std::cerr << "== LEGATE ERROR: " << exn.what() << "\n";
    std::cerr << "== LEGATE ERROR:\n";
    std::cerr << parser;
    std::exit(EXIT_FAILURE);
  }

  auto rt = Realm::Runtime::get_runtime();

  // ensure sensible utility
  if (const auto nutil = util.value.value(); nutil < 1) {
    throw TracedException<std::invalid_argument>{
      fmt::format("{} must be at least 1 (have {})", util.flag, nutil)};
  }

  autoconfigure(&rt, &util, &cpus, &gpus, &omps, &ompthreads, &sysmem, &fbmem, &numamem);

  if (Config::show_config) {
    // Can't use a logger, since Realm hasn't been initialized yet.
    std::cout << "Legate hardware configuration:";
    std::cout << " " << cpus.flag << "=" << cpus.value.raw_value();
    std::cout << " " << gpus.flag << "=" << gpus.value.raw_value();
    std::cout << " " << omps.flag << "=" << omps.value.raw_value();
    std::cout << " " << ompthreads.flag << "=" << ompthreads.value.raw_value();
    std::cout << " " << util.flag << "=" << util.value.raw_value();
    std::cout << " " << sysmem.flag << "=" << sysmem.value.raw_value();
    std::cout << " " << numamem.flag << "=" << numamem.value.raw_value();
    std::cout << " " << fbmem.flag << "=" << fbmem.value.raw_value();
    std::cout << " " << zcmem.flag << "=" << zcmem.value.raw_value();
    std::cout << " " << regmem.flag << "=" << regmem.value.raw_value();
    // Use of endl is deliberate, we want a newline and flush
    std::cout << std::endl;  // NOLINT(performance-avoid-endl)
  }

  // Set core configuration properties
  set_core_config_properties(&rt, parser, cpus, util, sysmem, regmem);

  // Set CUDA configuration properties
  set_cuda_config_properties(&rt, parser, gpus, fbmem, zcmem);

  // Set OpenMP configuration properties
  set_openmp_config_properties(&rt, parser, omps, ompthreads, numamem);

  set_legion_default_args(std::move(log_dir.value),
                          eager_alloc_percent.value,
                          std::move(log_levels.value),
                          profile.value,
                          spy.value,
                          freeze_on_error.value,
                          log_to_file.value);
}

}  // namespace legate::detail
