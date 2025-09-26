/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/parse.h>
#include <legate/runtime/detail/argument_parsing/util.h>
#include <legate/utilities/detail/env_defaults.h>

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <utilities/env.h>
#include <utilities/utilities.h>
#include <vector>

namespace test_parse_args {

class ParseArgsUnitNoEnv : public DefaultFixture {};

class ParseArgsUnit : public DefaultFixture {
  using Environment     = legate::test::Environment;
  using TemporaryEnvVar = legate::test::Environment::TemporaryEnvVar;

  TemporaryEnvVar legate_test_ = Environment::temporary_cleared_env_var("LEGATE_TEST");
  TemporaryEnvVar legate_auto_config_ =
    Environment::temporary_cleared_env_var("LEGATE_AUTO_CONFIG");
  TemporaryEnvVar legate_show_config_ =
    Environment::temporary_cleared_env_var("LEGATE_SHOW_CONFIG");
  TemporaryEnvVar legate_show_progress_ =
    Environment::temporary_cleared_env_var("LEGATE_SHOW_PROGRESS");
  TemporaryEnvVar legate_empty_task_ = Environment::temporary_cleared_env_var("LEGATE_EMPTY_TASK");
  TemporaryEnvVar legate_warmup_nccl_ =
    Environment::temporary_cleared_env_var("LEGATE_WARMUP_NCCL");
  TemporaryEnvVar legate_inline_task_launch_ =
    Environment::temporary_cleared_env_var("LEGATE_INLINE_TASK_LAUNCH");
  TemporaryEnvVar legate_show_usage_ = Environment::temporary_cleared_env_var("LEGATE_SHOW_USAGE");
  TemporaryEnvVar legate_max_exception_size_ =
    Environment::temporary_cleared_env_var("LEGATE_MAX_EXCEPTION_SIZE");
  TemporaryEnvVar legate_min_cpu_chunk_ =
    Environment::temporary_cleared_env_var("LEGATE_MIN_CPU_CHUNK");
  TemporaryEnvVar legate_min_gpu_chunk_ =
    Environment::temporary_cleared_env_var("LEGATE_MIN_GPU_CHUNK");
  TemporaryEnvVar legate_min_omp_chunk_ =
    Environment::temporary_cleared_env_var("LEGATE_MIN_OMP_CHUNK");
  TemporaryEnvVar legate_window_size_ =
    Environment::temporary_cleared_env_var("LEGATE_WINDOW_SIZE");
  TemporaryEnvVar legate_field_reuse_frac_ =
    Environment::temporary_cleared_env_var("LEGATE_FIELD_REUSE_FRAC");
  TemporaryEnvVar legate_field_reuse_freq_ =
    Environment::temporary_cleared_env_var("LEGATE_FIELD_REUSE_FREQ");
  TemporaryEnvVar legate_consensus_ = Environment::temporary_cleared_env_var("LEGATE_CONSENSUS");
  TemporaryEnvVar legate_disable_mpi_ =
    Environment::temporary_cleared_env_var("LEGATE_DISABLE_MPI");
  TemporaryEnvVar legate_io_use_vfd_gds_ =
    Environment::temporary_cleared_env_var("LEGATE_IO_USE_VFD_GDS");
  TemporaryEnvVar legate_log_mapping_ =
    Environment::temporary_cleared_env_var("LEGATE_LOG_MAPPING");
  TemporaryEnvVar legate_log_partitioning_ =
    Environment::temporary_cleared_env_var("LEGATE_LOG_PARTITIONING");
  TemporaryEnvVar legate_cuda_driver_ =
    Environment::temporary_cleared_env_var("LEGATE_CUDA_DRIVER");
};

MATCHER_P(
  ScaledArgumentMatches,  // NOLINT
  matcher,                // NOLINT
  fmt::format("{} a Scaled argument and the unscaled value {}",
              negation ? "isn't" : "is",
              ::testing::DescribeMatcher<typename std::decay_t<arg_type>::value_type::value_type>(
                matcher, negation)))
{
  return ::testing::ExplainMatchResult(matcher, arg.value().unscaled_value(), result_listener);
}

MATCHER_P(ArgumentMatches,  // NOLINT
          matcher,          // NOLINT
          fmt::format("{} an Argument and the value {}",
                      negation ? "isn't" : "is",
                      ::testing::DescribeMatcher<typename std::decay_t<arg_type>::value_type>(
                        matcher, negation)))  // NOLINT
{
  return ::testing::ExplainMatchResult(matcher, arg.value(), result_listener);
}

TEST_F(ParseArgsUnit, EmptyArgs)
{
  ASSERT_THAT([] { static_cast<void>(legate::detail::parse_args({})); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Command-line argument to parse must have at least 1 value")));
}

TEST_F(ParseArgsUnit, InvalidBooleanArgs)
{
  ASSERT_THAT(
    [] { static_cast<void>(legate::detail::parse_args({"dummy", "--auto-config", "yes1"})); },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Unknown boolean argument yes1, expected one of '1, t, true, y, yes' or "
                           "'0, f, false, n, no'")));
}

TEST_F(ParseArgsUnit, NoArgs)
{
  const auto parsed = legate::detail::parse_args({"dummy"});

  ASSERT_THAT(parsed.auto_config, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.show_config, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.show_progress, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.empty_task, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.warmup_nccl, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.inline_task_launch, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.show_usage, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.max_exception_size, ArgumentMatches(LEGATE_MAX_EXCEPTION_SIZE_DEFAULT));
  ASSERT_THAT(parsed.min_cpu_chunk, ArgumentMatches(LEGATE_MIN_CPU_CHUNK_DEFAULT));
  ASSERT_THAT(parsed.min_gpu_chunk, ArgumentMatches(LEGATE_MIN_GPU_CHUNK_DEFAULT));
  ASSERT_THAT(parsed.min_omp_chunk, ArgumentMatches(LEGATE_MIN_OMP_CHUNK_DEFAULT));
  ASSERT_THAT(parsed.window_size, ArgumentMatches(LEGATE_WINDOW_SIZE_DEFAULT));
  ASSERT_THAT(parsed.field_reuse_frac, ArgumentMatches(LEGATE_FIELD_REUSE_FRAC_DEFAULT));
  ASSERT_THAT(parsed.field_reuse_freq, ArgumentMatches(LEGATE_FIELD_REUSE_FREQ_DEFAULT));
  ASSERT_THAT(parsed.consensus, ArgumentMatches(bool{LEGATE_CONSENSUS_DEFAULT}));
  ASSERT_THAT(parsed.disable_mpi, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.io_use_vfd_gds, ArgumentMatches(legate::detail::is_gds_maybe_available()));
  ASSERT_THAT(parsed.cpus, ArgumentMatches(-1));
  ASSERT_THAT(parsed.gpus, ArgumentMatches(-1));
  ASSERT_THAT(parsed.omps, ArgumentMatches(-1));
  ASSERT_THAT(parsed.ompthreads, ArgumentMatches(-1));
  ASSERT_THAT(parsed.util, ArgumentMatches(2));
  ASSERT_THAT(parsed.sysmem, ScaledArgumentMatches(-1));
  ASSERT_THAT(parsed.numamem, ScaledArgumentMatches(-1));
  ASSERT_THAT(parsed.fbmem, ScaledArgumentMatches(-1));
  ASSERT_THAT(parsed.zcmem, ScaledArgumentMatches(128));
  ASSERT_THAT(parsed.regmem, ScaledArgumentMatches(0));
  ASSERT_THAT(parsed.profile, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.profile_name, ArgumentMatches(std::string{"legate"}));
  ASSERT_THAT(parsed.provenance, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.log_levels, ArgumentMatches(std::string{}));
  ASSERT_THAT(parsed.log_dir, ArgumentMatches(std::filesystem::current_path()));
  ASSERT_THAT(parsed.log_to_file, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.freeze_on_error, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.cuda_driver_path,
              ArgumentMatches(std::string{LEGATE_SHARED_LIBRARY_PREFIX
                                          "cuda" LEGATE_SHARED_LIBRARY_SUFFIX ".1"}));
}

TEST_F(ParseArgsUnitNoEnv, NoArgs)
{
#define TEMP_ENV_VAR(__var_name__, __var_value__)                       \
  const auto __var_name__ = legate::test::Environment::TemporaryEnvVar{ \
    LEGATE_STRINGIZE(__var_name__), LEGATE_STRINGIZE(__var_value__), /* overwrite */ true}

  TEMP_ENV_VAR(LEGATE_AUTO_CONFIG, 0);
  TEMP_ENV_VAR(LEGATE_SHOW_CONFIG, 1);
  TEMP_ENV_VAR(LEGATE_SHOW_PROGRESS, 1);
  TEMP_ENV_VAR(LEGATE_EMPTY_TASK, 1);
  TEMP_ENV_VAR(LEGATE_WARMUP_NCCL, LEGATE_DEFINED(LEGATE_USE_NCCL));
  TEMP_ENV_VAR(LEGATE_INLINE_TASK_LAUNCH, 1);
  TEMP_ENV_VAR(LEGATE_SHOW_USAGE, 1);
  TEMP_ENV_VAR(LEGATE_MAX_EXCEPTION_SIZE, 1234);
  TEMP_ENV_VAR(LEGATE_MIN_CPU_CHUNK, 1234);
  TEMP_ENV_VAR(LEGATE_MIN_GPU_CHUNK, 4321);
  TEMP_ENV_VAR(LEGATE_MIN_OMP_CHUNK, 6789);
  TEMP_ENV_VAR(LEGATE_WINDOW_SIZE, 42);
  TEMP_ENV_VAR(LEGATE_FIELD_REUSE_FRAC, 77);
  TEMP_ENV_VAR(LEGATE_FIELD_REUSE_FREQ, 88);
  TEMP_ENV_VAR(LEGATE_CONSENSUS, 1);
  TEMP_ENV_VAR(LEGATE_DISABLE_MPI, 1);
  TEMP_ENV_VAR(LEGATE_IO_USE_VFD_GDS, 0);
  TEMP_ENV_VAR(LEGATE_LOG_MAPPING, 1);
  TEMP_ENV_VAR(LEGATE_LOG_PARTITIONING, 1);
  TEMP_ENV_VAR(LEGATE_CUDA_DRIVER, libdummy_cuda_driver.so);

  const auto parsed = legate::detail::parse_args({"dummy"});

  ASSERT_THAT(parsed.auto_config, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.show_config, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.show_progress, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.empty_task, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.warmup_nccl, ArgumentMatches(bool{LEGATE_DEFINED(LEGATE_USE_NCCL)}));
  ASSERT_THAT(parsed.inline_task_launch, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.show_usage, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.max_exception_size, ArgumentMatches(1234));
  ASSERT_THAT(parsed.min_cpu_chunk, ArgumentMatches(1234));
  ASSERT_THAT(parsed.min_gpu_chunk, ArgumentMatches(4321));
  ASSERT_THAT(parsed.min_omp_chunk, ArgumentMatches(6789));
  ASSERT_THAT(parsed.window_size, ArgumentMatches(42));
  ASSERT_THAT(parsed.field_reuse_frac, ArgumentMatches(77));
  ASSERT_THAT(parsed.field_reuse_freq, ArgumentMatches(88));
  ASSERT_THAT(parsed.consensus, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.disable_mpi, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.io_use_vfd_gds, ArgumentMatches(legate::detail::is_gds_maybe_available()));
  ASSERT_THAT(parsed.cpus, ArgumentMatches(-1));
  ASSERT_THAT(parsed.gpus, ArgumentMatches(-1));
  ASSERT_THAT(parsed.omps, ArgumentMatches(-1));
  ASSERT_THAT(parsed.ompthreads, ArgumentMatches(-1));
  ASSERT_THAT(parsed.util, ArgumentMatches(2));
  ASSERT_THAT(parsed.sysmem, ScaledArgumentMatches(-1));
  ASSERT_THAT(parsed.numamem, ScaledArgumentMatches(-1));
  ASSERT_THAT(parsed.fbmem, ScaledArgumentMatches(-1));
  ASSERT_THAT(parsed.zcmem, ScaledArgumentMatches(128));
  ASSERT_THAT(parsed.regmem, ScaledArgumentMatches(0));
  ASSERT_THAT(parsed.profile, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.profile_name, ArgumentMatches(std::string{"legate"}));
  ASSERT_THAT(parsed.provenance, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.log_levels,
              ArgumentMatches(std::string{"legate.mapper=info,legate.partitioner=debug"}));
  ASSERT_THAT(parsed.log_dir, ArgumentMatches(std::filesystem::current_path()));
  ASSERT_THAT(parsed.log_to_file, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.freeze_on_error, ArgumentMatches(::testing::IsFalse()));
  ASSERT_THAT(parsed.cuda_driver_path, ArgumentMatches(std::string{"libdummy_cuda_driver.so"}));

#undef TEMP_ENV_VAR
}

using ParseArgsDeathTest = ParseArgsUnit;

TEST_F(ParseArgsDeathTest, InvalidArgs)
{
  ASSERT_EXIT(static_cast<void>(legate::detail::parse_args({"dummy", "--invalid-args"})),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              ::testing::HasSubstr("== LEGATE ERROR: Unknown argument: --invalid-args"));
}

namespace {

class BoolCLArgs {
 public:
  template <typename T>
  std::string operator()(const ::testing::TestParamInfo<T>& info) const;

  [[nodiscard]] static auto values();
};

template <typename T>
std::string BoolCLArgs::operator()(const ::testing::TestParamInfo<T>& info) const
{
  return fmt::format("{}_{}", info.param.first, info.param.second);
}

auto BoolCLArgs::values()
{
  static constexpr std::pair<std::string_view, bool> values[] = {
    // All falsey values
    std::make_pair("0", false),
    std::make_pair("f", false),
    std::make_pair("False", false),
    std::make_pair("n", false),
    std::make_pair("No", false),
    // All truthy values
    std::make_pair("", true),
    std::make_pair("1", true),
    std::make_pair("t", true),
    std::make_pair("True", true),
    std::make_pair("y", true),
    std::make_pair("Yes", true),
  };

  return ::testing::ValuesIn(values);
}

}  // namespace

class BoolArgs : public ParseArgsUnit,
                 public ::testing::WithParamInterface<std::pair<std::string_view, bool>> {};

INSTANTIATE_TEST_SUITE_P(ParseArgsUnit, BoolArgs, BoolCLArgs::values(), BoolCLArgs{});

TEST_P(BoolArgs, AutoConfig)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--auto-config", std::string{arg_value}});

  ASSERT_THAT(parsed.auto_config, ArgumentMatches(expected));
}

TEST_P(BoolArgs, ShowConfig)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--show-config", std::string{arg_value}});

  ASSERT_THAT(parsed.show_config, ArgumentMatches(expected));
}

TEST_P(BoolArgs, ShowProgress)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--show-progress", std::string{arg_value}});

  ASSERT_THAT(parsed.show_progress, ArgumentMatches(expected));
}

TEST_P(BoolArgs, EmptyTask)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--use-empty-task", std::string{arg_value}});

  ASSERT_THAT(parsed.empty_task, ArgumentMatches(expected));
}

TEST_P(BoolArgs, WarmupNCCL)
{
  const auto [arg_value, expected] = GetParam();
  auto args = std::vector<std::string>{"dummy", "--warmup-nccl", std::string{arg_value}};

  // If --warmup-nccl is passed (with a truthy value) on a config that doesn't have support for
  // NCCL, then we throw an exception. So this handling is a little gnarly...
  if constexpr (LEGATE_DEFINED(LEGATE_USE_NCCL)) {
    const auto parsed = legate::detail::parse_args(std::move(args));

    ASSERT_THAT(parsed.warmup_nccl, ArgumentMatches(expected));
  } else if (expected) {
    // We are on a system WITHOUT NCCL and expect to warmup NCCL. This should fail.
    ASSERT_THAT([&] { static_cast<void>(legate::detail::parse_args(std::move(args))); },
                ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
                  "Cannot warmup NCCL, Legate was not configured with NCCL support")));
  } else {
    // We are on a system WITHOUT NCCL and do NOT expect to warm it up. This is OK.
    const auto parsed = legate::detail::parse_args(std::move(args));

    ASSERT_THAT(parsed.warmup_nccl, ArgumentMatches(::testing::IsFalse()));
  }
}

TEST_P(BoolArgs, InlineTaskLaunch)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--inline-task-launch", std::string{arg_value}});

  ASSERT_THAT(parsed.inline_task_launch, ArgumentMatches(expected));
}

TEST_P(BoolArgs, ShowUsage)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--show-memory-usage", std::string{arg_value}});

  ASSERT_THAT(parsed.show_usage, ArgumentMatches(expected));
}

TEST_F(ParseArgsUnit, MaxExceptionSize)
{
  constexpr auto MAGIC = 1234;
  const auto parsed =
    legate::detail::parse_args({"dummy", "--max-exception-size", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.max_exception_size, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, MinCPUChunk)
{
  constexpr auto MAGIC = 333;
  const auto parsed =
    legate::detail::parse_args({"dummy", "--min-cpu-chunk", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.min_cpu_chunk, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, MinGPUChunk)
{
  constexpr auto MAGIC = 333;
  const auto parsed =
    legate::detail::parse_args({"dummy", "--min-gpu-chunk", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.min_gpu_chunk, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, MinOMPChunk)
{
  constexpr auto MAGIC = 333;
  const auto parsed =
    legate::detail::parse_args({"dummy", "--min-omp-chunk", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.min_omp_chunk, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, WindowSize)
{
  constexpr auto MAGIC = 42;
  const auto parsed = legate::detail::parse_args({"dummy", "--window-size", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.window_size, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, FieldReuseFrac)
{
  constexpr auto MAGIC = 42;
  const auto parsed =
    legate::detail::parse_args({"dummy", "--field-reuse-fraction", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.field_reuse_frac, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, FieldReuseFreq)
{
  constexpr auto MAGIC = 42;
  const auto parsed =
    legate::detail::parse_args({"dummy", "--field-reuse-frequency", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.field_reuse_freq, ArgumentMatches(MAGIC));
}

TEST_P(BoolArgs, Consensus)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed = legate::detail::parse_args({"dummy", "--consensus", std::string{arg_value}});

  ASSERT_THAT(parsed.consensus, ArgumentMatches(expected));
}

TEST_P(BoolArgs, DisableMPI)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--disable-mpi", std::string{arg_value}});

  ASSERT_THAT(parsed.disable_mpi, ArgumentMatches(expected));
}

TEST_P(BoolArgs, IOUseVFDGDS)
{
  const auto [arg_value, expected] = GetParam();
  auto args = std::vector<std::string>{"dummy", "--io-use-vfd-gds", std::string{arg_value}};

  // If --io-use-vfd-gds is passed (with a truthy value) on a config that doesn't have support
  // for VFD-GDS, then we throw an exception. So this handling is a little gnarly...
  if constexpr (LEGATE_DEFINED(LEGATE_USE_HDF5_VFD_GDS)) {
    const auto parsed = legate::detail::parse_args(std::move(args));

    ASSERT_THAT(parsed.io_use_vfd_gds, ArgumentMatches(expected));
  } else if (expected) {
    // We are on a system WITHOUT VFD-GDS and expect to use it. This should fail.
    ASSERT_THAT([&] { static_cast<void>(legate::detail::parse_args(std::move(args))); },
                ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
                  "Cannot enable HDF5 VFD GDS, Legate was not configured with GDS support.")));
  } else {
    // We are on a system WITHOUT VFD GDS and do NOT expect to use it. This is OK.
    const auto parsed = legate::detail::parse_args(std::move(args));

    ASSERT_THAT(parsed.io_use_vfd_gds, ArgumentMatches(::testing::IsFalse()));
  }
}

TEST_F(ParseArgsUnit, CPUs)
{
  constexpr auto MAGIC = 42;
  const auto parsed    = legate::detail::parse_args({"dummy", "--cpus", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.cpus, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, CPUsInvalid)
{
  ASSERT_THAT([] { static_cast<void>(legate::detail::parse_args({"dummy", "--cpus", "-1"})); },
              ::testing::ThrowsMessage<std::out_of_range>(
                ::testing::HasSubstr("Number of CPU cores must be >=0, have -1")));
}

TEST_F(ParseArgsUnit, GPUs)
{
  constexpr auto MAGIC = 42;
  const auto parsed    = legate::detail::parse_args({"dummy", "--gpus", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.gpus, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, GPUsInvalid)
{
  ASSERT_THAT([] { static_cast<void>(legate::detail::parse_args({"dummy", "--gpus", "-1"})); },
              ::testing::ThrowsMessage<std::out_of_range>(
                ::testing::HasSubstr("Number of GPUs must be >=0, have -1")));
}

TEST_F(ParseArgsUnit, OMPs)
{
  constexpr auto MAGIC = 42;
  const auto parsed    = legate::detail::parse_args({"dummy", "--omps", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.omps, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, OMPsInvalid)
{
  ASSERT_THAT([] { static_cast<void>(legate::detail::parse_args({"dummy", "--omps", "-1"})); },
              ::testing::ThrowsMessage<std::out_of_range>(
                ::testing::HasSubstr("Number of OpenMP groups must be >=0, have -1")));
}

TEST_F(ParseArgsUnit, OMPThreads)
{
  constexpr auto MAGIC = 42;
  const auto parsed = legate::detail::parse_args({"dummy", "--ompthreads", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.ompthreads, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, OMPThreadsInvalid)
{
  ASSERT_THAT(
    [] { static_cast<void>(legate::detail::parse_args({"dummy", "--ompthreads", "-1"})); },
    ::testing::ThrowsMessage<std::out_of_range>(
      ::testing::HasSubstr("Number of threads per OpenMP group must be >=0, have -1")));
}

TEST_F(ParseArgsUnit, Util)
{
  constexpr auto MAGIC = 42;
  const auto parsed    = legate::detail::parse_args({"dummy", "--utility", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.util, ArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, UtilInvalid)
{
  ASSERT_THAT([] { static_cast<void>(legate::detail::parse_args({"dummy", "--utility", "0"})); },
              ::testing::ThrowsMessage<std::out_of_range>(
                ::testing::HasSubstr("Number of utility threads must be >0, have 0")));
}

TEST_F(ParseArgsUnit, Sysmem)
{
  constexpr auto MAGIC = 20;
  const auto parsed    = legate::detail::parse_args({"dummy", "--sysmem", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.sysmem, ScaledArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, NUMAmem)
{
  constexpr auto MAGIC = 20;
  const auto parsed    = legate::detail::parse_args({"dummy", "--numamem", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.numamem, ScaledArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, FBmem)
{
  constexpr auto MAGIC = 20;
  const auto parsed    = legate::detail::parse_args({"dummy", "--fbmem", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.fbmem, ScaledArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, ZCmem)
{
  constexpr auto MAGIC = 20;
  const auto parsed    = legate::detail::parse_args({"dummy", "--zcmem", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.zcmem, ScaledArgumentMatches(MAGIC));
}

TEST_F(ParseArgsUnit, Regmem)
{
  constexpr auto MAGIC = 20;
  const auto parsed    = legate::detail::parse_args({"dummy", "--regmem", std::to_string(MAGIC)});

  ASSERT_THAT(parsed.regmem, ScaledArgumentMatches(MAGIC));
}

TEST_P(BoolArgs, Profile)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed = legate::detail::parse_args({"dummy", "--profile", std::string{arg_value}});

  ASSERT_THAT(parsed.profile, ArgumentMatches(expected));
}

TEST_F(ParseArgsUnit, ProfileEnabled)
{
  const auto parsed = legate::detail::parse_args({"dummy", "--profile", "t"});

  ASSERT_THAT(parsed.profile, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.profile_name, ArgumentMatches(std::string{"legate"}));
  ASSERT_THAT(parsed.log_levels, ArgumentMatches(std::string{"legion_prof=info"}));
}

TEST_F(ParseArgsUnit, ProfileEnabledWithFilename)
{
  const auto parsed =
    legate::detail::parse_args({"dummy", "--profile", "t", "--profile-name", "foo"});

  ASSERT_THAT(parsed.profile, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.profile_name, ArgumentMatches(std::string{"foo"}));
  ASSERT_THAT(parsed.log_levels, ArgumentMatches(std::string{"legion_prof=info"}));
}

TEST_F(ParseArgsUnit, ProfileEnabledAndArgs)
{
  const auto parsed =
    legate::detail::parse_args({"dummy", "--profile", "t", "--logging", "foo=bar"});

  ASSERT_THAT(parsed.profile, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.log_levels, ArgumentMatches(std::string{"foo=bar,legion_prof=info"}));
}

TEST_F(ParseArgsUnit, ProvenanceEnabled)
{
  const auto parsed = legate::detail::parse_args({"dummy", "--provenance", "t"});

  ASSERT_THAT(parsed.provenance, ArgumentMatches(::testing::IsTrue()));
}

TEST_F(ParseArgsUnit, ProvenanceEnabledAndArgs)
{
  const auto parsed = legate::detail::parse_args({"dummy", "--provenance", "t", "--profile", "f"});

  ASSERT_THAT(parsed.provenance, ArgumentMatches(::testing::IsTrue()));
  ASSERT_THAT(parsed.profile, ArgumentMatches(::testing::IsFalse()));
}

TEST_F(ParseArgsUnit, LogLevels)
{
  const auto levels = std::string{"foo=bar,baz=bop"};
  const auto parsed = legate::detail::parse_args({"dummy", "--logging", levels});

  ASSERT_THAT(parsed.log_levels, ArgumentMatches(levels));
}

TEST_F(ParseArgsUnit, LogDir)
{
  const auto path   = std::filesystem::path{"foo"} / "bar" / "baz";
  const auto parsed = legate::detail::parse_args({"dummy", "--logdir", path});

  ASSERT_THAT(parsed.log_dir, ArgumentMatches(path));
}

TEST_P(BoolArgs, LogToFile)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--log-to-file", std::string{arg_value}});

  ASSERT_THAT(parsed.log_to_file, ArgumentMatches(expected));
}

TEST_P(BoolArgs, FreezeOnError)
{
  const auto [arg_value, expected] = GetParam();
  const auto parsed =
    legate::detail::parse_args({"dummy", "--freeze-on-error", std::string{arg_value}});

  ASSERT_THAT(parsed.freeze_on_error, ArgumentMatches(expected));
}

TEST_F(ParseArgsUnit, CUDADriverPath)
{
  const auto path   = std::string{"/path/to/cuda/driver.so"};
  const auto parsed = legate::detail::parse_args({"dummy", "--cuda-driver-path", path});

  ASSERT_THAT(parsed.cuda_driver_path, ArgumentMatches(path));
}

TEST_F(ParseArgsUnit, Deduplication)
{
  const auto orig = std::vector<std::string>{"dummy",
                                             "--cpus=1",
                                             "--cpus",
                                             "1",
                                             "--log-to-file=f",
                                             "--gpus=0",
                                             "--profile",
                                             "--cpus",
                                             "2",
                                             "--gpus=0"};
  const auto expected =
    std::vector<std::string>{"dummy", "--log-to-file=f", "--profile", "--cpus", "2", "--gpus=0"};
  const auto dedup = legate::detail::deduplicate_command_line_flags(orig);

  ASSERT_THAT(dedup, ::testing::ContainerEq(expected));
}

}  // namespace test_parse_args
