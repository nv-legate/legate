/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/parse.h>

#include <legate/mapping/detail/base_mapper.h>
#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/flags/logging.h>
#include <legate/runtime/detail/argument_parsing/util.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/env_defaults.h>
#include <legate/utilities/detail/string_utils.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/typedefs.h>
#include <legate/version.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <argparse/argparse.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>

namespace {

template <typename T>
class ArgView {
 public:
  const legate::detail::Argument<T>& arg;
};

template <typename T>
ArgView(const legate::detail::Argument<T>&) -> ArgView<T>;

}  // namespace

namespace fmt {

template <typename T>
struct formatter<ArgView<T>> : public formatter<std::string_view> {
  format_context::iterator format(const ArgView<T>& view, format_context& ctx) const
  {
    return format_to(ctx.out(), "{} = {}", view.arg.name(), view.arg.value());
  }
};

template <typename T>
struct formatter<ArgView<legate::detail::Scaled<T>>> : public formatter<std::string_view> {
  format_context::iterator format(const ArgView<legate::detail::Scaled<T>>& view,
                                  format_context& ctx) const
  {
    return format_to(ctx.out(),
                     "{} = {} ({} {})",
                     view.arg.name(),
                     fmt::group_digits(view.arg.value().scaled_value()),
                     fmt::group_digits(view.arg.value().unscaled_value()),
                     view.arg.value().unit());
  }
};

}  // namespace fmt

namespace legate::detail {

std::string ParsedArgs::config_summary() const
{
  auto ret             = std::string{};
  const auto print_var = [&](const auto& var) {
    fmt::format_to(std::back_inserter(ret), "- {}\n", ArgView{var});
  };

  fmt::format_to(std::back_inserter(ret),
                 "==============================================\n"
                 "Legate configuration summary:\n");
  print_var(auto_config);
  print_var(show_progress);
  print_var(empty_task);
  print_var(warmup_nccl);
  print_var(inline_task_launch);
  // No point printing this one, obviously we are showing usage
  // print_var(show_usage);
  print_var(max_exception_size);
  print_var(min_cpu_chunk);
  print_var(min_gpu_chunk);
  print_var(min_omp_chunk);
  print_var(window_size);
  print_var(field_reuse_frac);
  print_var(field_reuse_freq);
  print_var(consensus);
  print_var(disable_mpi);
  print_var(io_use_vfd_gds);
  print_var(cpus);
  print_var(gpus);
  print_var(ompthreads);
  print_var(util);
  print_var(sysmem);
  print_var(numamem);
  print_var(fbmem);
  print_var(zcmem);
  print_var(regmem);
  print_var(profile);
  print_var(profile_name);
  print_var(provenance);
  print_var(log_levels);
  print_var(log_dir);
  print_var(log_to_file);
  print_var(freeze_on_error);
  print_var(cuda_driver_path);
  ret += "==============================================";
  return ret;
}

// ==========================================================================================

namespace {

class LegateArgumentParser {
 public:
  LegateArgumentParser();

  template <typename T>
  [[nodiscard]] Argument<T> add_argument(std::string flag, std::string help, T init);

  template <typename T>
  [[nodiscard]] Argument<Scaled<T>> add_scaled_argument(std::string flag,
                                                        std::string help,
                                                        Scaled<T> init);

  void parse_args(std::vector<std::string> args) const;

  [[nodiscard]] const std::shared_ptr<argparse::ArgumentParser>& parser() const;

 private:
  template <typename T>
  void add_argument_common_(std::string_view flag,
                            std::string help,
                            const T& default_value,
                            T& dest_value);

  [[nodiscard]] std::vector<std::string> argparse_bool_flag_workaround_(
    std::vector<std::string> args) const;
  [[nodiscard]] std::vector<std::string> argparse_duplicate_flag_workaround_(
    Span<const std::string> args) const;

  std::shared_ptr<argparse::ArgumentParser> parser_{};
  std::unordered_set<std::string_view> bool_flags_{};
};

// ------------------------------------------------------------------------------------------

class ParseBoolArg {
 public:
  explicit ParseBoolArg(bool& dest);

  [[nodiscard]] bool operator()(std::string_view value) const;

 private:
  [[nodiscard]] static bool do_parse_(std::string_view value);

  std::reference_wrapper<bool> dest_;
};

ParseBoolArg::ParseBoolArg(bool& dest) : dest_{dest} {}

bool ParseBoolArg::operator()(std::string_view value) const
{
  dest_.get() = do_parse_(value);
  return dest_;
}

bool ParseBoolArg::do_parse_(std::string_view value)
{
  if (value.empty()) {
    // The implicit value, which is true
    return true;
  }

  const auto equal_to_value = [lo = string_to_lower(std::string{value})](std::string_view val) {
    return val == lo;
  };
  constexpr std::string_view truthy_values[] = {"1", "t", "true", "y", "yes"};
  constexpr std::string_view falsey_values[] = {"0", "f", "false", "n", "no"};

  if (std::any_of(std::begin(truthy_values), std::end(truthy_values), equal_to_value)) {
    return true;
  }
  if (std::any_of(std::begin(falsey_values), std::end(falsey_values), equal_to_value)) {
    return false;
  }

  throw TracedException<std::invalid_argument>{
    fmt::format("Unknown boolean argument {}, expected one of '{}' or '{}'",
                value,
                fmt::join(truthy_values, ", "),
                fmt::join(falsey_values, ", "))};
}

// ------------------------------------------------------------------------------------------

template <typename T>
void LegateArgumentParser::add_argument_common_(std::string_view flag,
                                                std::string help,
                                                const T& default_value,
                                                T& dest_value)
{
  auto& parg = parser()
                 ->add_argument(flag)
                 .help(std::move(help))
                 .default_value(default_value)
                 .store_into(dest_value);

  using RawT = std::decay_t<T>;

  if constexpr (std::is_same_v<RawT, bool>) {
    parg.metavar("BOOL")
      .implicit_value(true)
      .nargs(argparse::nargs_pattern::optional)
      .action(ParseBoolArg{dest_value});
  } else {
    parg.nargs(1);
    if constexpr (std::is_integral_v<RawT>) {
      parg.metavar("INT");
    } else if constexpr (std::is_floating_point_v<RawT>) {
      parg.metavar("FLOAT");
    } else if constexpr (std::is_same_v<RawT, std::string>) {
      parg.metavar("STRING");
    } else if constexpr (std::is_same_v<RawT, std::filesystem::path>) {
      parg.metavar("PATH");
    }
  }
}

std::vector<std::string> LegateArgumentParser::argparse_bool_flag_workaround_(
  std::vector<std::string> args) const
{
  // argparse does not handle boolean flags that have an optional value correctly. Until this
  // is fixed, we need to manually traverse the argument list and convert any solitary flags:
  //
  // --flag --some-option value --flag
  //
  // to
  //
  // --flag 1 --some-option value --flag 1
  //
  // See https://github.com/p-ranav/argparse/issues/404
  for (auto it = args.begin(); it != args.end(); ++it) {
    if (bool_flags_.find(*it) == bool_flags_.end()) {
      continue;
    }

    const auto next_it = std::next(it);

    if (next_it == args.end()) {
      // --flag is the last parameter in the flags list
      args.emplace_back("1");
      break;
    }

    // Check if the next entry is a flag (which all start with '-'), use rfind() to mimic
    // startswith(). See https://stackoverflow.com/a/40441240
    if (next_it->rfind('-', 0) == 0) {
      it = args.insert(next_it, "1");
    }
  }
  return args;
}

std::vector<std::string> LegateArgumentParser::argparse_duplicate_flag_workaround_(
  Span<const std::string> args) const
{
  // argparse does not handle duplicate flags! It complains with: "Duplicate flag --flag", so
  // we need to convert
  //
  // --flag 1 --some-option value --flag 2
  //
  // to
  //
  // --some-option value --flag 2
  //
  // See https://github.com/p-ranav/argparse/issues/404
  return deduplicate_command_line_flags(args);
}

// ------------------------------------------------------------------------------------------

LegateArgumentParser::LegateArgumentParser()
  : parser_{std::make_shared<argparse::ArgumentParser>(
      /* program_name */ "LEGATE_CONFIG can contain:",
      /* version */ fmt::format(
        "{}.{}.{}", LEGATE_VERSION_MAJOR, LEGATE_VERSION_MINOR, LEGATE_VERSION_PATCH))}
{
  constexpr auto MAX_LINE_WIDTH = 80;

  parser()->set_usage_max_line_width(MAX_LINE_WIDTH);
}

template <typename T>
Argument<T> LegateArgumentParser::add_argument(std::string flag, std::string help, T init)
{
  auto ret = Argument<T>{parser(), std::move(flag), std::move(init)};

  add_argument_common_(ret.flag(), std::move(help), ret.value(), ret.value_mut());
  if constexpr (std::is_same_v<std::decay_t<T>, bool>) {
    bool_flags_.insert(ret.flag());
  }
  return ret;
}

template <typename T>
Argument<Scaled<T>> LegateArgumentParser::add_scaled_argument(std::string flag,
                                                              std::string help,
                                                              Scaled<T> init)
{
  auto ret = Argument<Scaled<T>>{parser(), std::move(flag), std::move(init)};

  add_argument_common_(ret.flag(),
                       std::move(help),
                       ret.value().unscaled_value(),
                       ret.value_mut().unscaled_value_mut());
  return ret;
}

void LegateArgumentParser::parse_args(std::vector<std::string> args) const
{
  args = argparse_bool_flag_workaround_(std::move(args));
  args = argparse_duplicate_flag_workaround_(args);

  try {
    parser()->parse_args(args);
  } catch (const TracedExceptionBase&) {
    // If it's one of our exceptions, rethrow it
    throw;
  } catch (const std::exception& exn) {
    std::cerr << "== LEGATE ERROR:\n";
    std::cerr << "== LEGATE ERROR: " << exn.what() << "\n";
    std::cerr << "== LEGATE ERROR:\n";
    std::cerr << *parser();
    std::exit(EXIT_FAILURE);
  }
}

const std::shared_ptr<argparse::ArgumentParser>& LegateArgumentParser::parser() const
{
  return parser_;
}

}  // namespace

ParsedArgs parse_args(std::vector<std::string> args)
{
  // argparse expects a argv-like set of arguments where the first entry is the program
  // name. It does not properly handle empty arguments and will segfault.
  if (args.empty()) {
    throw TracedException<std::invalid_argument>{
      "Command-line argument to parse must have at least 1 value"};
  }

  // values with -1 defaults will be auto-configured via the Realm API
  constexpr std::int32_t DEFAULT_CPUS       = -1;
  constexpr std::int32_t DEFAULT_GPUS       = -1;
  constexpr std::int32_t DEFAULT_OMPS       = -1;
  constexpr std::int32_t DEFAULT_OMPTHREADS = -1;
  constexpr std::int32_t DEFAULT_UTILITY    = 2;
  constexpr std::int64_t DEFAULT_SYSMEM     = -1;
  constexpr std::int64_t DEFAULT_NUMAMEM    = -1;
  constexpr std::int64_t DEFAULT_FBMEM      = -1;
  constexpr std::int64_t DEFAULT_ZCMEM      = 128;  // MB
  constexpr std::int64_t DEFAULT_REGMEM     = 0;    // MB
  constexpr auto MB                         = 1 << 20;

  auto parser = LegateArgumentParser{};

  // ------------------------------------------------------------------------------------------
  parser.parser()->add_group("Core configuration");

  auto auto_config = parser.add_argument(
    "--auto-config",
    "Automatically detect a suitable configuration.\n"
    "\n"
    "If enabled, attempts to detect a reasonable default for most options listed hereafter and is "
    "recommended for most users. If disabled, Legate will use a minimal set of resources.",
    LEGATE_AUTO_CONFIG.get().value_or(true));
  auto show_config =
    parser.add_argument("--show-config",
                        "Print the configuration to stdout.\n"
                        "\n"
                        "The configuration is shown after all auto-configuration has been done, "
                        "and so is representative of the 'final' configuration used. This variable "
                        "can be used to visually confirm that Legate's automatic configuration "
                        "heuristics are picking up appropriate settings for your machine.",
                        LEGATE_SHOW_CONFIG.get().value_or(false));
  auto show_usage =
    parser.add_argument("--show-memory-usage",
                        "Show detailed memory footprint of Legate at program shutdown.",
                        LEGATE_SHOW_USAGE.get().value_or(false));
  auto show_progress = parser.add_argument(
    "--show-progress",
    "Print a progress summary before each task is executed.\n"
    "\n"
    "This is useful to visually ensure that a particular task is being called. The progress "
    "reports are emitted by Legate before entering into the task body itself.",
    LEGATE_SHOW_PROGRESS.get().value_or(false));

  // ------------------------------------------------------------------------------------------
  parser.parser()->add_group("Resource configuration");

  auto cpus = parser.add_argument(
    "--cpus",
    "Number of standalone CPU cores to reserve per rank, must be >=0.\n"
    "\n"
    "Setting this to a value > 1 will enable additional parallelism in addition to multi-node "
    "execution. For example, running a program across 4 processes with --cpus 2 will result in a "
    "total of 2 * 4 = 8 parallel threads of execution.\n"
    "\n"
    "For best performance, the number of reserved CPU cores should not exceed the number of "
    "physical cores on the CPU (hyper-threaded physical cores typically count as 2 logical cores), "
    "as Legate will attempt to uniquely pin a thread to each reserved core.\n"
    "\n"
    "If the system does not support thread pinning, it is unspecified which cores a thread will "
    "execute on. If the system does support thread pinning, but the number of requested CPUs "
    "exceed the number of cores, then Legate will oversubscribe the available cores. The manner in "
    "which the cores are oversubscribed is unspecified.",
    DEFAULT_CPUS);

  cpus.action([](std::string_view, const Argument<std::int32_t>* cpus_arg) {
    if (cpus_arg->value() < 0) {
      throw TracedException<std::out_of_range>{
        fmt::format("Number of CPU cores must be >=0, have {}", cpus_arg->value())};
    }
    return cpus_arg->value();
  });

  auto gpus =
    parser.add_argument("--gpus", "Number of GPUs to reserve per rank, must be >=0.", DEFAULT_GPUS);

  gpus.action([](std::string_view, const Argument<std::int32_t>* gpus_arg) {
    if (gpus_arg->value() < 0) {
      throw TracedException<std::out_of_range>{
        fmt::format("Number of GPUs must be >=0, have {}", gpus_arg->value())};
    }
    return gpus_arg->value();
  });

  auto omps = parser.add_argument(
    "--omps",
    "Number of OpenMP groups to use per rank, must be >=0.\n"
    "\n"
    "Each OpenMP group reserves --ompthreads number of CPU physical cores. For example, running a "
    "program across 4 processes, with --omps 2 --ompthreads 4 will result in a combined total of 4 "
    "* 2 * 4 = 32 parallel threads of execution.\n"
    "\n"
    "OpenMP cores are distinct from (and additional to) CPU cores reserved via --cpus, and for the "
    "purposes of core reservations, count as separate cores. The behavior for pinning and "
    "oversubscription is identical to that of regular CPU cores.",
    DEFAULT_OMPS);

  omps.action([](std::string_view, const Argument<std::int32_t>* omps_arg) {
    if (omps_arg->value() < 0) {
      throw TracedException<std::out_of_range>{
        fmt::format("Number of OpenMP groups must be >=0, have {}", omps_arg->value())};
    }
    return omps_arg->value();
  });

  auto ompthreads =
    parser.add_argument("--ompthreads",
                        "Number of threads (reserved CPU cores) per OpenMP group, must be >=0",
                        DEFAULT_OMPTHREADS);

  ompthreads.action([](std::string_view, const Argument<std::int32_t>* ompthreads_arg) {
    if (ompthreads_arg->value() < 0) {
      throw TracedException<std::out_of_range>{fmt::format(
        "Number of threads per OpenMP group must be >=0, have {}", ompthreads_arg->value())};
    }
    return ompthreads_arg->value();
  });

  auto util = parser.add_argument(
    "--utility",
    "Number of threads per rank to use for runtime meta-work, must be >0.\n"
    "\n"
    "Legate will attempt to reserve a unique core per utility thread if possible. Failing this, "
    "Legate will attempt to spread the utility threads out over any remaining unclaimed cores. "
    "While utility threads may oversubscribe cores assigned to other utility threads, they may not "
    "cohabitate with compute threads (--cpus/--omps). Thus, when manually configuring "
    "reservations, the user must ensure that at least one unclaimed core remains to place the "
    "utility threads.\n"
    "\n"
    "For best performance, it is recommended that utility threads each reserve a unique core.",
    DEFAULT_UTILITY);

  util.action([](std::string_view, const Argument<std::int32_t>* util_arg) {
    if (util_arg->value() < 1) {
      throw TracedException<std::out_of_range>{
        fmt::format("Number of utility threads must be >0, have {}", util_arg->value())};
    }
    return util_arg->value();
  });

  // ------------------------------------------------------------------------------------------
  parser.parser()->add_group("Memory allocation");

  auto sysmem  = parser.add_scaled_argument("--sysmem",
                                           "Size (in MiB) of DRAM memory to reserve per rank",
                                           Scaled{DEFAULT_SYSMEM, MB, "MiB"});
  auto numamem = parser.add_scaled_argument(
    "--numamem",
    "Size (in MiB) of NUMA-specific DRAM memory to reserve per NUMA domain",
    Scaled{DEFAULT_NUMAMEM, MB, "MiB"});
  auto fbmem = parser.add_scaled_argument(
    "--fbmem",
    "Size (in MiB) of GPU (or \"framebuffer\") memory to reserve per GPU",
    Scaled{DEFAULT_FBMEM, MB, "MiB"});
  auto zcmem = parser.add_scaled_argument(
    "--zcmem",
    "Size (in MiB) of GPU-registered (or \"zero-copy\") DRAM memory to reserve per GPU",
    Scaled{DEFAULT_ZCMEM, MB, "MiB"});
  auto regmem             = parser.add_scaled_argument("--regmem",
                                           "Size (in MiB) of NIC-registered DRAM memory to reserve",
                                           Scaled{DEFAULT_REGMEM, MB, "MiB"});
  auto max_exception_size = parser.add_argument(
    "--max-exception-size",
    "Maximum size (in bytes) to allocate for exception messages.\n"
    "\n"
    "Legate needs an upper bound on the size of exception that can be raised by a task.",
    LEGATE_MAX_EXCEPTION_SIZE.get(LEGATE_MAX_EXCEPTION_SIZE_DEFAULT,
                                  LEGATE_MAX_EXCEPTION_SIZE_TEST));

  auto min_cpu_chunk = parser.add_argument(
    "--min-cpu-chunk",
    "Minimum CPU chunk size (in bytes).\n"
    "\n"
    "If using CPUs, any task operating on arrays smaller than this will not be parallelized across "
    "more than one core.",
    LEGATE_MIN_CPU_CHUNK.get(LEGATE_MIN_CPU_CHUNK_DEFAULT, LEGATE_MIN_CPU_CHUNK_TEST));
  auto min_gpu_chunk = parser.add_argument(
    "--min-gpu-chunk",
    "Minimum GPU chunk size (in bytes).\n"
    "\n"
    "If using GPUs, any task operating on arrays smaller than this will not be parallelized across "
    "more than one core.",
    LEGATE_MIN_GPU_CHUNK.get(LEGATE_MIN_GPU_CHUNK_DEFAULT, LEGATE_MIN_GPU_CHUNK_TEST));
  auto min_omp_chunk = parser.add_argument(
    "--min-omp-chunk",
    "Minimum OpenMP chunk size (in bytes)."
    "\n"
    "If using OpenMP, any task operating on arrays smaller than this will not be parallelized "
    "across more than one OpenMP group.",
    LEGATE_MIN_OMP_CHUNK.get(LEGATE_MIN_OMP_CHUNK_DEFAULT, LEGATE_MIN_OMP_CHUNK_TEST));

  // ------------------------------------------------------------------------------------------
  parser.parser()->add_group("Execution control");

  auto window_size = parser.add_argument(
    "--window-size",
    "Maximum size of the submitted operation queue before forced flush.",
    LEGATE_WINDOW_SIZE.get(LEGATE_WINDOW_SIZE_DEFAULT, LEGATE_WINDOW_SIZE_TEST));
  auto warmup_nccl = parser.add_argument(
    "--warmup-nccl",
    "Perform a warmup for NCCL on startup.\n"
    "\n"
    "NCCL usually has a relatively high startup cost the first time any collective communication "
    "is performed. This could corrupt performance measurements if that startup is performed in the "
    "hot-path. This is useful when doing performance benchmarks.",
    LEGATE_WARMUP_NCCL.get().value_or(false));

  warmup_nccl.action([](std::string_view, const Argument<bool>* warmup_nccl_arg) {
    const auto val = warmup_nccl_arg->value();

    if (val && !LEGATE_DEFINED(LEGATE_USE_NCCL)) {
      throw TracedException<std::runtime_error>{
        "Cannot warmup NCCL, Legate was not configured with NCCL support."};
    }
    return val;
  });

  auto disable_mpi = parser.add_argument(
    "--disable-mpi",
    "Disable MPI initialization and use.\n"
    "\n"
    "This is useful if Legate was configured with MPI support (which usually causes Legate to use "
    "it), but MPI is not functional on the current system. When this flag is passed, no task "
    "should be launched that requests the MPI communicator, or the program will fail.",
    LEGATE_DISABLE_MPI.get(LEGATE_DISABLE_MPI_DEFAULT, LEGATE_DISABLE_MPI_TEST));

  auto field_reuse_frac = parser.add_argument(
    "--field-reuse-fraction",
    "What amount (in bytes) of the \"primary\" memory type should be allocated before consensus "
    "match is triggered.\n"
    "\n"
    "Which memory is \"primary\" depends on the configuration of Legate. If Legate was configured "
    "for GPU support (and it detects GPUs), then the GPU memory is primary. If Legate is "
    "configured for OpenMP support, then socket memory is considered primary.\n"
    "\n"
    "For example, on a hypothetical machine with 1,000 bytes of GPU memory, and 500 bytes of "
    "socket memory, a value of --field-reuse-fraction=250 will result in Legate issuing a "
    "consensus match every 250 bytes of GPU memory being allocated."
    "\n"
    "See help string of --consensus for more information on what a consensus match constitutes.",
    LEGATE_FIELD_REUSE_FRAC.get(LEGATE_FIELD_REUSE_FRAC_DEFAULT, LEGATE_FIELD_REUSE_FRAC_TEST));
  auto field_reuse_freq = parser.add_argument(
    "--field-reuse-frequency",
    "The size (in number of stores) of the discarded store/array field cache to retain.\n"
    "\n"
    "When Legate stores and arrays are destroyed, their backing storage is not immediately "
    "deallocated. Instead, it is first sent to a cache to be potentially reused when another store "
    "of similar properties is requested. This flag sets the maximum size of this cache.\n"
    "\n"
    "Higher values may result in faster execution (as more stores may be constructed out of the "
    "cache) at the tradeoff of higher memory usage.",
    LEGATE_FIELD_REUSE_FREQ.get(LEGATE_FIELD_REUSE_FREQ_DEFAULT, LEGATE_FIELD_REUSE_FREQ_TEST));
  auto consensus = parser.add_argument(
    "--consensus",
    "Whether to perform the RegionField consensus match operation on single-node runs (for "
    "testing).\n"
    "\n"
    "This is normally only necessary on multi-node runs, where all processes must collectively "
    "agree that a RegionField has been garbage collected at the Python level before it can be "
    "reused.",
    LEGATE_CONSENSUS.get(LEGATE_CONSENSUS_DEFAULT, LEGATE_CONSENSUS_TEST));

  auto inline_task_launch = parser.add_argument(
    "--inline-task-launch",
    "Enable inline task launch.\n"
    "\n"
    "Normally, when a task is launched, Legate goes through the \"Legion calling convention\" "
    "which involves serialization of all arguments, and packaging up the task such that it may be "
    "handed off to Legion for later execution. Crucially, this later execution may happen on "
    "another thread, possibly on another node.\n"
    "\n"
    "However, for single-processor runs, this process is both overly costly and largely "
    "unnecessary. For example, there is no need to perform any partitioning analysis, as -- by "
    "virtue of being single-processor -- the data will be used in full. In such cases it may be "
    "profitable to launch the tasks directly on the same processor/thread which submitted them, "
    "i.e. \"inline\".\n"
    "\n"
    "Note that enabling this mode will constrain execution to a single processor, even if more  "
    "are available.\n"
    "\n"
    "This feature is currently marked experimental, and should not be relied upon. The current "
    "implementation is not guaranteed to always be profitable. It may offer dramatic speedup in "
    "some circumstances, but it may also lead to large slowdowns in others. Future improvements "
    "will seek to improve this, at which point it will be moved to the normal Legate namespace.",
    experimental::LEGATE_INLINE_TASK_LAUNCH.get().value_or(false));

  inline_task_launch.argparse_argument().hidden();

  // ------------------------------------------------------------------------------------------
  parser.parser()->add_group("Profiling and logging");

  auto profile      = parser.add_argument("--profile", "Whether to collect profiling logs", false);
  auto profile_name = parser.add_argument(
    "--profile-name",
    "Base filename for profiling logs\n"
    "\n"
    "Will create one file per rank (<profile-name>_<rank>.prof) relative to the log directory.",
    std::string{"legate"});
  auto provenance = parser.add_argument(
    "--provenance",
    "Whether to record call provenance. \n"
    "\n"
    "Enabling call provenance will cause stack trace information to be included in Legion "
    "profiles, progress output, nvtx ranges, and some error messages. Enabling --profile "
    "will automatically enable --provenance.",
    false);
  auto log_levels  = parser.add_argument("--logging", logging_help_str(), std::string{});
  auto log_dir     = parser.add_argument("--logdir",
                                     "Directory to emit logfiles to, defaults to current directory",
                                     std::filesystem::current_path());
  auto log_to_file = parser.add_argument(
    "--log-to-file", "Redirect logging output to a file inside --logdir", false);

  // ------------------------------------------------------------------------------------------
  parser.parser()->add_group("Debugging");

  auto freeze_on_error = parser.add_argument(
    "--freeze-on-error",
    "If the program crashes, freeze execution right before exit so a debugger can be attached",
    false);
  auto empty_task =
    parser.add_argument("--use-empty-task",
                        "Execute an empty dummy task in place of each task execution.\n"
                        "\n"
                        "This is primarily a developer feature for use in debugging runtime or "
                        "scheduling inconsistencies and is not recommended for external use.",
                        LEGATE_EMPTY_TASK.get().value_or(false));

  // ------------------------------------------------------------------------------------------
  parser.parser()->add_group("Miscellaneous options");

  auto cuda_driver_path = parser.add_argument(
    "--cuda-driver-path",
    "Path to the CUDA driver shared library object\n"
    "\n"
    "The given path can either be:\n"
    "1. An absolute path: The path is used as-is to load the shared object at that location.\n"
    "2. A relative path: The path is used as-is, but the directory against which the relative\n"
    "   lookup is implementation defined. Most implementations however will look up the object\n"
    "   relative to the current working directory.\n"
    "3. A name: The shared object lookup is completely implementation defined. If on Linux, see\n"
    "   dlopen(3) for a complete description on the lookup mechanism in this case.\n"
    "\n"
    "The user should generally not need to set this variable, but it can be useful in case the "
    "driver needs to be interposed by a user-supplied shim.",
    LEGATE_CUDA_DRIVER.get().value_or(
      // "libcuda.so.1" on Linux
      LEGATE_SHARED_LIBRARY_PREFIX "cuda" LEGATE_SHARED_LIBRARY_SUFFIX ".1"));
  auto io_use_vfd_gds =
    parser.add_argument("--io-use-vfd-gds",
                        "Whether to enable HDF5 Virtual File Driver (VDS) GPUDirectStorage (GDS) "
                        "which may dramatically speed up file storage and extraction",
                        LEGATE_IO_USE_VFD_GDS.get().value_or(is_gds_maybe_available()));

  io_use_vfd_gds.action([](std::string_view, const Argument<bool>* io_use_vfd_gds_arg) {
    const auto val = io_use_vfd_gds_arg->value();

    if (val && !LEGATE_DEFINED(LEGATE_USE_HDF5_VFD_GDS)) {
      throw TracedException<std::runtime_error>{
        "Cannot enable HDF5 VFD GDS, Legate was not configured with GDS support."};
    }
    return val;
  });

  auto experimental_copy_path = parser.add_argument(
    "--experimental-copy-path",
    "Enable conditional copy optimizations based on workload characteristics.\n"

    "This feature is currently marked experimental, and should not be relied upon. The current "
    "implementation may offer performance improvements in some circumstances, but it may also "
    "lead to slowdowns in others. Future improvements will seek to optimize this further.",
    false);

  experimental_copy_path.argparse_argument().hidden();

  parser.parse_args(std::move(args));

  const auto add_logger = [&](std::string_view logger, std::string_view level = "info") {
    auto& lvls = log_levels.value_mut();

    if (!lvls.empty()) {
      lvls += ',';
    }
    lvls += logger;
    lvls += '=';
    lvls += level;
  };

  if (LEGATE_LOG_MAPPING.get().value_or(false)) {
    add_logger(mapping::detail::BaseMapper::LOGGER_NAME);
  }

  if (LEGATE_LOG_PARTITIONING.get().value_or(false)) {
    add_logger(log_legate_partitioner().get_name(), "debug");
  }

  if (profile.value()) {
    add_logger("legion_prof");
  }

  return {/* auto_config */ std::move(auto_config),
          /* show_config */ std::move(show_config),
          /* show_progress */ std::move(show_progress),
          /* empty_task */ std::move(empty_task),
          /* warmup_nccl */ std::move(warmup_nccl),
          /* inline_task_launch */ std::move(inline_task_launch),
          /* show_usage */ std::move(show_usage),
          /* max_exception_size */ std::move(max_exception_size),
          /* min_cpu_chunk */ std::move(min_cpu_chunk),
          /* min_gpu_chunk */ std::move(min_gpu_chunk),
          /* min_omp_chunk */ std::move(min_omp_chunk),
          /* window_size */ std::move(window_size),
          /* field_reuse_frac */ std::move(field_reuse_frac),
          /* field_reuse_freq */ std::move(field_reuse_freq),
          /* consensus */ std::move(consensus),
          /* disable_mpi */ std::move(disable_mpi),
          /* io_use_vfd_gds */ std::move(io_use_vfd_gds),
          /* cpus */ std::move(cpus),
          /* gpus */ std::move(gpus),
          /* omps */ std::move(omps),
          /* ompthreads */ std::move(ompthreads),
          /* util */ std::move(util),
          /* sysmem */ std::move(sysmem),
          /* numamem */ std::move(numamem),
          /* fbmem */ std::move(fbmem),
          /* zcmem */ std::move(zcmem),
          /* regmem */ std::move(regmem),
          /* profile */ std::move(profile),
          /* profile_name */ std::move(profile_name),
          /* provenance */ std::move(provenance),
          /* log_levels */ std::move(log_levels),
          /* log_dir */ std::move(log_dir),
          /* log_to_file */ std::move(log_to_file),
          /* freeze_on_error */ std::move(freeze_on_error),
          /* cuda_driver_path */ std::move(cuda_driver_path),
          /* experimental_copy_path */ std::move(experimental_copy_path)};
}

}  // namespace legate::detail
