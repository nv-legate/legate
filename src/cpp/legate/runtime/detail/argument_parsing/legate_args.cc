/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/legate_args.h>

#include <legate/runtime/detail/argument_parsing/config_legion.h>
#include <legate/runtime/detail/argument_parsing/config_realm.h>
#include <legate/runtime/detail/argument_parsing/flags/cpus.h>
#include <legate/runtime/detail/argument_parsing/flags/cuda_driver_path.h>
#include <legate/runtime/detail/argument_parsing/flags/fbmem.h>
#include <legate/runtime/detail/argument_parsing/flags/gpus.h>
#include <legate/runtime/detail/argument_parsing/flags/numamem.h>
#include <legate/runtime/detail/argument_parsing/flags/ompthreads.h>
#include <legate/runtime/detail/argument_parsing/flags/openmp.h>
#include <legate/runtime/detail/argument_parsing/flags/sysmem.h>
#include <legate/runtime/detail/argument_parsing/parse.h>
#include <legate/runtime/detail/argument_parsing/util.h>
#include <legate/runtime/detail/config.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/string_utils.h>
#include <legate/utilities/detail/traced_exception.h>

#include <realm/runtime.h>

#include <fmt/core.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace legate::detail {

namespace {

std::string LEGATE_CONFIG_SAVED{};

}  // namespace

std::string_view get_parsed_LEGATE_CONFIG() { return LEGATE_CONFIG_SAVED; }

namespace {

[[nodiscard]] ParsedArgs parse_legate_args()
{
  LEGATE_CONFIG_SAVED = LEGATE_CONFIG.get().value_or("");

  auto args = string_split(LEGATE_CONFIG_SAVED);

  // Needed to satisfy argparse, which expects an argv-like interface where argv[0] is the
  // program name.
  args.insert(args.begin(), "LEGATE_CONFIG");
  return parse_args(std::move(args));
}

Config prefill_config(const ParsedArgs& args)
{
  Config cfg;

  cfg.set_auto_config(args.auto_config.value());
  cfg.set_show_config(args.show_config.value());
  cfg.set_show_progress_requested(args.show_progress.value());
  cfg.set_use_empty_task(args.empty_task.value());
  cfg.set_warmup_nccl(args.warmup_nccl.value());
  cfg.set_enable_inline_task_launch(args.inline_task_launch.value());
  cfg.set_show_mapper_usage(args.show_usage.value());
  cfg.set_max_exception_size(args.max_exception_size.value());
  cfg.set_min_cpu_chunk(args.min_cpu_chunk.value());
  cfg.set_min_gpu_chunk(args.min_gpu_chunk.value());
  cfg.set_min_omp_chunk(args.min_omp_chunk.value());
  cfg.set_window_size(args.window_size.value());
  cfg.set_field_reuse_frac(args.field_reuse_frac.value());
  cfg.set_field_reuse_freq(args.field_reuse_freq.value());
  cfg.set_consensus(args.consensus.value());
  cfg.set_disable_mpi(args.disable_mpi.value());
  cfg.set_io_use_vfd_gds(args.io_use_vfd_gds.value());
  // Disable MPI in legate if the network bootstrap is p2p
  if (REALM_UCP_BOOTSTRAP_MODE.get() == "p2p") {
    cfg.set_disable_mpi(true);
    cfg.set_need_network(true);
  } else {
    cfg.set_need_network(multi_node_job());
  }
  cfg.set_profile_name(args.profile_name.value());
  // Turn on provenance if profiling is enabled
  cfg.set_profile(args.profile.value());
  if (args.provenance.was_set()) {
    cfg.set_provenance(args.provenance.value());
  } else {
    cfg.set_provenance(args.profile.value());
  }
  return cfg;
}

void autoconfigure(ParsedArgs* args, Config* config)
{
  auto&& rt = Realm::Runtime::get_runtime();

  // We can hold core by reference, as it is required to exist
  const auto& core_mod = [&rt] {
    const auto* mod = rt.get_module_config("core");

    LEGATE_CHECK(mod != nullptr);
    return *mod;
  }();

  const auto auto_config = config->auto_config();
  const auto* cuda_mod   = rt.get_module_config("cuda");

  // This must happen before --fbmem is called, as we need to have initialized the CUDA driver
  // API before then.
  configure_cuda_driver_path(args->cuda_driver_path);
  // auto-configure --gpus
  configure_gpus(auto_config, cuda_mod, &args->gpus, config);
  // auto-configure --fbmem
  configure_fbmem(auto_config, cuda_mod, args->gpus, &args->fbmem);

  std::vector<std::size_t> numa_mems{};

  if (const auto* numa_mod = rt.get_module_config("numa")) {
    if (numa_mod->get_resource("numa_mems", numa_mems) != REALM_SUCCESS) {
      // Some kind of error happened during NUMA detection, so pretend like we don't have any
      // NUMA memory
      numa_mems.clear();
    }
  }

  // auto-configure --omps
  configure_omps(auto_config, rt.get_module_config("openmp"), numa_mems, args->gpus, &args->omps);
  // auto-configure --numamem
  configure_numamem(auto_config, numa_mems, args->omps, &args->numamem);
  // auto-configure --sysmem
  configure_sysmem(auto_config, core_mod, args->numamem, &args->sysmem);
  // auto-configure --cpus
  configure_cpus(auto_config, core_mod, args->omps, args->util, args->gpus, &args->cpus);
  // auto-configure --ompthreads
  configure_ompthreads(auto_config,
                       core_mod,
                       args->util,
                       args->cpus,
                       args->gpus,
                       args->omps,
                       &args->ompthreads,
                       config);
}

}  // namespace

Config handle_legate_args()
{
  auto parsed = parse_legate_args();
  auto cfg    = prefill_config(parsed);

  autoconfigure(&parsed, &cfg);
  if (cfg.show_config()) {
    // Can't use a logger, since Realm hasn't been initialized yet.
    fmt::println(parsed.config_summary());
  }

  // These config flags are set by the autoconfigure call above, but allow the user to be able
  // to see the configuration before doing these checks. That way, if they are confused, they
  // can pass `--show-config` to see what Legate determined.
  if (!LEGATE_DEFINED(LEGATE_USE_CUDA) && cfg.need_cuda()) {
    throw TracedException<std::runtime_error>{
      "Legate was run with GPUs but was not built with GPU support. Please "
      "install Legate again with the \"--with-cuda\" flag"};
  }
  if (!LEGATE_DEFINED(LEGATE_USE_OPENMP) && cfg.need_openmp()) {
    throw TracedException<std::runtime_error>{
      "Legate was run with OpenMP enabled, but was not built with OpenMP "
      "support. Please install Legate again with the \"--with-openmp\" "
      "flag"};
  }

  configure_realm(parsed);
  configure_legion(parsed);
  return cfg;
}

}  // namespace legate::detail
