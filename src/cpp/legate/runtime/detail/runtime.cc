/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/runtime.h>

#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/coll.h>
#include <legate/comm/detail/comm.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/detail/external_allocation.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/logical_store_partition.h>
#include <legate/data/detail/physical_store.h>
#include <legate/experimental/io/detail/task.h>
#include <legate/mapping/detail/core_mapper.h>
#include <legate/mapping/detail/default_mapper.h>
#include <legate/mapping/detail/machine.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/operation/detail/fill.h>
#include <legate/operation/detail/offload.h>
#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/redop/detail/register.h>
#include <legate/runtime/detail/argument_parsing/legate_args.h>
#include <legate/runtime/detail/argument_parsing/util.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/mpi_detection.h>
#include <legate/runtime/resource.h>
#include <legate/runtime/runtime.h>
#include <legate/task/detail/legion_task.h>
#include <legate/task/detail/task_info.h>
#include <legate/task/task_config.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/tuning/scope.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/env_defaults.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/linearize.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/hash.h>
#include <legate/utilities/scope_guard.h>

#include <realm/cuda/cuda_module.h>
#include <realm/module_config.h>
#include <realm/network.h>
#include <realm/runtime.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <atomic>
#include <cstdlib>
#include <mappers/logging_wrapper.h>
#include <nccl.h>
// GCC 14 alloc-zero warning when using Conda installed compiler
LEGATE_PRAGMA_PUSH();
LEGATE_PRAGMA_GCC_IGNORE("-Walloc-zero");
#include <regex>
LEGATE_PRAGMA_POP();
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace legate::detail {

Logger& log_legate()
{
  static Logger log{"legate"};

  return log;
}

Logger& log_legate_partitioner()
{
  static Logger log{"legate.partitioner"};

  return log;
}

namespace {

// This is the unique string name for our library which can be used from both C++ and Python to
// generate IDs
constexpr std::string_view CORE_LIBRARY_NAME = "legate.core";

}  // namespace

Runtime::Runtime(Config config) : config_{std::move(config)}, scope_{this->config()}
{
  static_cast<void>(scope_.exchange_scheduling_window_size(this->config().window_size()));
}

Library& Runtime::create_library(
  std::string_view library_name,
  const ResourceConfig& config,
  std::unique_ptr<mapping::Mapper> mapper,
  // clang-tidy complains that this is not moved (but clearly, it is). I suspect this is due to
  // a tiny little footnote in try_emplace():
  //
  // ...unlike [unordered_map::]insert() or [unordered_map::]emplace(), these functions do not
  // move from rvalue arguments if the insertion does not happen...
  //
  // So probably there is a branch that clang-tidy sees down in try_emplace() that does
  // this. But clang-tidy seemingly does not realize that the emplacement always happens,
  // because we first check that the key does not yet exist.
  //
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  std::map<VariantCode, VariantOptions> default_options)
{
  if (libraries_.find(library_name) != libraries_.end()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Library {} already exists", library_name)};
  }

  log_legate().debug() << "Library " << library_name << " is created";
  if (nullptr == mapper) {
    mapper = std::make_unique<mapping::detail::DefaultMapper>();
  }

  return libraries_
    .try_emplace(std::string{library_name},
                 std::string{library_name},
                 config,
                 std::move(mapper),
                 std::move(default_options))
    .first->second;
}

namespace {

template <typename T, typename U>
using const_like_t = std::conditional_t<std::is_const_v<T>, const U, U>;

template <typename LibraryMapT>
[[nodiscard]] auto find_library_impl(LibraryMapT& libraries, std::string_view library_name)
  -> std::optional<std::reference_wrapper<const_like_t<LibraryMapT, Library>>>
{
  const auto it = libraries.find(library_name);

  if (libraries.end() == it) {
    return std::nullopt;
  }
  return it->second;
}

}  // namespace

std::optional<std::reference_wrapper<const Library>> Runtime::find_library(
  std::string_view library_name) const
{
  return find_library_impl(libraries_, library_name);
}

std::optional<std::reference_wrapper<Library>> Runtime::find_library(std::string_view library_name)
{
  return find_library_impl(libraries_, library_name);
}

Library& Runtime::find_or_create_library(
  std::string_view library_name,
  const ResourceConfig& config,
  std::unique_ptr<mapping::Mapper> mapper,
  const std::map<VariantCode, VariantOptions>& default_options,
  bool* created)
{
  auto result = find_library(library_name);

  if (result.has_value()) {
    if (created) {
      *created = false;
    }
    return *result;
  }
  result = create_library(std::move(library_name), config, std::move(mapper), default_options);
  if (created) {
    *created = true;
  }
  return *result;
}

void Runtime::record_reduction_operator(std::uint32_t type_uid,
                                        std::int32_t op_kind,
                                        GlobalRedopID legion_op_id)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Record reduction op (type_uid: " << type_uid
                         << ", op_kind: " << op_kind
                         << ", legion_op_id: " << fmt::underlying(legion_op_id) << ")";
  }

  const auto inserted = reduction_ops_.try_emplace({type_uid, op_kind}, legion_op_id).second;

  if (!inserted) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Reduction op {} already exists for type {}", op_kind, type_uid)};
  }
}

GlobalRedopID Runtime::find_reduction_operator(std::uint32_t type_uid, std::int32_t op_kind) const
{
  const auto finder = reduction_ops_.find({type_uid, op_kind});

  if (reduction_ops_.end() == finder) {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_legate().debug() << "Can't find reduction op (type_uid: " << type_uid
                           << ", op_kind: " << op_kind << ")";
    }
    throw TracedException<std::invalid_argument>{
      fmt::format("Reduction op {} does not exist for type {}", op_kind, type_uid)};
  }
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Found reduction op " << fmt::underlying(finder->second)
                         << " (type_uid: " << type_uid << ", op_kind: " << op_kind << ")";
  }
  return finder->second;
}

std::pair<mapping::detail::Machine, const VariantInfo&> Runtime::slice_machine_for_task_(
  const TaskInfo& info) const
{
  const auto& machine = get_machine();
  auto sliced         = machine.only_if([&](mapping::TaskTarget t) {
    return info.find_variant(mapping::detail::to_variant_code(t)).has_value();
  });

  if (sliced.empty()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Task {} does not have any valid variant for the current "
                  "machine configuration {}",
                  info,
                  machine)};
  }

  const auto variant = info.find_variant(sliced.preferred_variant());

  LEGATE_ASSERT(variant.has_value());
  return {sliced, *variant};  // NOLINT(bugprone-unchecked-optional-access)
}

// This function should be moved to the library context
InternalSharedPtr<AutoTask> Runtime::create_task(const Library& library, LocalTaskID task_id)
{
  auto&& [machine, vinfo] = slice_machine_for_task_(*library.find_task(task_id));
  return make_internal_shared<AutoTask>(
    library, vinfo, task_id, new_op_id(), scope().priority(), std::move(machine));
}

InternalSharedPtr<ManualTask> Runtime::create_task(const Library& library,
                                                   LocalTaskID task_id,
                                                   const Domain& launch_domain)
{
  if (launch_domain.empty()) {
    throw TracedException<std::invalid_argument>{"Launch domain must not be empty"};
  }
  auto&& [machine, vinfo] = slice_machine_for_task_(*library.find_task(task_id));
  return make_internal_shared<ManualTask>(
    library, vinfo, task_id, launch_domain, new_op_id(), scope().priority(), std::move(machine));
}

void Runtime::issue_fill(InternalSharedPtr<LogicalStore> lhs, InternalSharedPtr<LogicalStore> value)
{
  if (lhs->unbound()) {
    throw TracedException<std::invalid_argument>{"Fill lhs cannot be an unbound store"};
  }
  if (value->volume() != 1) {
    throw TracedException<std::invalid_argument>{"Fill value should have only one element"};
  }

  submit(make_internal_shared<Fill>(
    std::move(lhs), std::move(value), new_op_id(), scope().priority(), get_machine()));
}

void Runtime::issue_fill(InternalSharedPtr<LogicalStore> lhs, Scalar value)
{
  if (lhs->unbound()) {
    throw TracedException<std::invalid_argument>{"Fill lhs cannot be an unbound store"};
  }

  submit(make_internal_shared<Fill>(
    std::move(lhs), std::move(value), new_op_id(), scope().priority(), get_machine()));
}

void Runtime::flush_scheduling_window(bool) {}

void Runtime::schedule_(std::deque<InternalSharedPtr<Operation>>* window)
{
  // We should only execute the operations resident in the queue at the time of flush, no more,
  // no less. It is possible that during the flush additional operations may be queued, and so
  // we use the size of the window on entrance to determine the number of ops to flush.
  //
  // It is also possible to invoke a recursive scheduling window flush, (for example, if an
  // operation launches a discard in its destructor, and we have a scheduling window of size
  // 1), so we must also guard against the possibility that a previous (recursive) flush has
  // already emptied the queue for us.
  //
  // This is why we check flush_size > 0 && !operations_.empty().
  for (auto flush_size = window->size(); flush_size > 0 && !window->empty(); --flush_size) {
    auto op = std::move(window->front());

    window->pop_front();

    if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_legate().debug() << op->to_string(true /*show_provenance*/) << " launched";
    }

    if (!op->needs_partitioning()) {
      op->launch();
      continue;
    }

    // TODO(wonchanl): We need the side effect from the launch calls to get key partitions set
    // correctly. In the future, the partitioner should manage key partitions.
    auto strategy = Partitioner::partition_stores(op.get());

    op->launch(&strategy);
  }
}

void Runtime::clear_scheduling_window(PrivateKey) { operations_.clear(); }

void Runtime::submit(InternalSharedPtr<Operation> op)
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << op->to_string(true /*show_provenance*/) << " submitted";
  }

  // This call will check the task signatures if they are set, so call it before appending to
  // operations_.
  op->validate();

  if (!op->needs_partitioning()) {
    op->launch();
    return;
  }

  // TODO(wonchanl): We need the side effect from the launch calls to get key partitions set
  // correctly. In the future, the partitioner should manage key partitions.
  auto strategy = Partitioner::partition_stores(op.get());

  op->launch(&strategy);
}

namespace {

void validate_store_shape(const InternalSharedPtr<Shape>& shape,
                          const InternalSharedPtr<Type>& type)
{
  if (shape->unbound()) {
    throw TracedException<std::invalid_argument>{
      "Shape of an unbound store cannot be used to create another store "
      "until the store is initialized by a task"};
  }
  if (type->variable_size()) {
    throw TracedException<std::invalid_argument>{"Store must have a fixed-size type"};
  }
}

}  // namespace

// Reserving the right to make this non-const in the future
// NOLINTNEXTLINE(readability-make-member-function-const)
InternalSharedPtr<LogicalStore> Runtime::create_store(InternalSharedPtr<Type> type,
                                                      std::uint32_t dim)
{
  check_dimensionality_(dim);
  auto storage = make_internal_shared<detail::Storage>(
    make_internal_shared<Shape>(dim), type->size(), get_provenance().as_string_view());
  return make_internal_shared<LogicalStore>(std::move(storage), std::move(type));
}  // namespace

// shape can be unbound in this function, so we shouldn't use the same validation as the other
// variants
// Reserving the right to make this non-const in the future
// NOLINTNEXTLINE(readability-make-member-function-const)
InternalSharedPtr<LogicalStore> Runtime::create_store(InternalSharedPtr<Shape> shape,
                                                      InternalSharedPtr<Type> type)
{
  if (type->variable_size()) {
    throw TracedException<std::invalid_argument>{"Store must have a fixed-size type"};
  }
  check_dimensionality_(shape->ndim());
  auto storage = make_internal_shared<detail::Storage>(
    std::move(shape), type->size(), get_provenance().as_string_view());
  return make_internal_shared<LogicalStore>(std::move(storage), std::move(type));
}

// Reserving the right to make this non-const in the future
// NOLINTNEXTLINE(readability-make-member-function-const)
InternalSharedPtr<LogicalStore> Runtime::create_store(const Scalar& scalar,
                                                      InternalSharedPtr<Shape> shape)
{
  validate_store_shape(shape, scalar.type());
  if (shape->volume() != 1) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Scalar stores must have a shape of volume 1, instead got shape {} of volume {}",
                  *shape,
                  shape->volume())};
  }
  auto storage = make_internal_shared<detail::Storage>(
    std::move(shape), scalar, get_provenance().as_string_view());
  return make_internal_shared<detail::LogicalStore>(std::move(storage), scalar.type());
}

InternalSharedPtr<LogicalStore> Runtime::create_store(
  const InternalSharedPtr<Shape>& shape,
  InternalSharedPtr<Type> type,
  InternalSharedPtr<ExternalAllocation> allocation,
  InternalSharedPtr<mapping::detail::DimOrdering> /*ordering*/)
{
  validate_store_shape(shape, type);
  LEGATE_CHECK(allocation->ptr());
  if (allocation->size() < shape->volume() * type->size()) {
    throw TracedException<std::invalid_argument>{fmt::format(
      "External allocation of size {} is not big enough for a store of shape {} and type {}",
      allocation->size(),
      shape->extents(),
      *type)};
  }

  if (config().enable_inline_task_launch()) {
    // Create an allocated Storage directly from the external allocation.
    auto storage = make_internal_shared<detail::Storage>(
      shape, type, allocation, get_provenance().as_string_view());
    return make_internal_shared<detail::LogicalStore>(std::move(storage), std::move(type));
  }

  auto store = create_store(shape, std::move(type));

  // TODO(wonchanl): Need a v2 implementation

  return store;
}

Runtime::IndexAttachResult Runtime::create_store(
  const InternalSharedPtr<Shape>& shape,
  const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& tile_shape,
  InternalSharedPtr<Type> type,
  Span<const std::pair<legate::ExternalAllocation, tuple<std::uint64_t>>> /*allocations*/,
  InternalSharedPtr<mapping::detail::DimOrdering> /*ordering*/)
{
  if (config().enable_inline_task_launch()) {
    throw TracedException<std::runtime_error>{
      "create_store with multiple external allocations is not supported with inline task launch"};
  }

  validate_store_shape(shape, type);

  auto store     = create_store(shape, std::move(type));
  auto partition = partition_store_by_tiling(store, tile_shape);

  // TODO(wonchanl): Need a v2 implementation
  return {std::move(store), std::move(partition)};
}

void Runtime::check_dimensionality_(std::uint32_t dim)
{
  if (dim > LEGATE_MAX_DIM) {
    throw TracedException<std::out_of_range>{
      fmt::format("The maximum number of dimensions is {}, but a {}-D store is requested",
                  LEGION_MAX_DIM,
                  dim)};
  }
}

void Runtime::issue_mapping_fence() {}

void Runtime::issue_execution_fence(bool)
{
  if (config().need_cuda()) {
    auto&& api = cuda::detail::get_cuda_driver_api();

    api->stream_synchronize(current_stream_);
  }
}

InternalSharedPtr<LogicalStore> Runtime::get_timestamp(Timing::Precision precision)
{
  static auto empty_shape =
    make_internal_shared<Shape>(SmallVector<std::uint64_t, LEGATE_MAX_DIM>{});
  auto timestamp = create_store(empty_shape, int64());
  submit(make_internal_shared<Timing>(new_op_id(), precision, timestamp));
  return timestamp;
}

InternalSharedPtr<mapping::detail::Machine> Runtime::create_toplevel_machine()
{
  const auto num_nodes = local_machine_.total_nodes;
  const auto num_gpus  = local_machine_.total_gpu_count();
  const auto num_omps  = local_machine_.total_omp_count();
  const auto num_cpus  = local_machine_.total_cpu_count();
  auto create_range    = [&num_nodes](std::uint32_t num_procs) {
    auto per_node_count = static_cast<std::uint32_t>(num_procs / num_nodes);
    return mapping::ProcessorRange{/*low_id=*/0, num_procs, per_node_count};
  };

  return make_internal_shared<mapping::detail::Machine>(
    std::map<mapping::TaskTarget, mapping::ProcessorRange>{
      {mapping::TaskTarget::GPU, create_range(num_gpus)},
      {mapping::TaskTarget::OMP, create_range(num_omps)},
      {mapping::TaskTarget::CPU, create_range(num_cpus)}});
}

const mapping::detail::Machine& Runtime::get_machine() const { return *scope().machine(); }

ZStringView Runtime::get_provenance() const { return scope().provenance(); }

namespace {

#ifdef REALM_USE_GASNETEX

[[nodiscard]] std::uint64_t gpus_per_rank_for_gasnetex_obcount()
{
#if !LEGATE_DEFINED(LEGATE_USE_CUDA)
  return 0;
#else
  auto&& realm = Realm::Runtime::get_runtime();

  int argc    = 0;
  char** argv = nullptr;
  // Realm #399 allows this early config pass; REALM_DEFAULT_ARGS is consumed during
  // Legion initialization, so GASNet-EX obcount must be sized before that point.
  if (!realm.create_configs(argc, argv)) {
    throw TracedException<std::runtime_error>{
      "Realm failed to create configs while sizing GASNet obcount."};
  }

  const auto* cuda = realm.get_module_config("cuda");
  if (cuda == nullptr) {
    return 0;
  }

  std::int32_t gpus = 0;
  if (const auto status = cuda->get_resource("gpu", gpus); status != REALM_SUCCESS) {
    throw TracedException<std::runtime_error>{fmt::format(
      "Realm CUDA module failed to report GPU resources (status {}).", static_cast<int>(status))};
  }
  return static_cast<std::uint64_t>(gpus);
#endif
}

#endif  // REALM_USE_GASNETEX

void handle_realm_default_args(bool need_network_init)
{
  constexpr EnvironmentVariable<std::string> REALM_DEFAULT_ARGS{"REALM_DEFAULT_ARGS"};
  std::stringstream ss;
#ifdef REALM_USE_GASNETEX
  bool user_set_gasnetex_obcount = false;
#endif

  if (const auto existing_default_args = REALM_DEFAULT_ARGS.get();
      existing_default_args.has_value()) {
    // If Realm sees multiple networks arguments, with one of them being "none", (e.g. "-ll:networks
    // foo -ll:networks none", or even "-ll:networks none -ll:networks none"), it balks with:
    //
    // "Cannot specify both 'none' and another value in -ll:networks"
    //
    // So we must strip away any existing -ll:networks arguments before we append our -ll:networks
    // argument.
#ifdef REALM_USE_GASNETEX
    user_set_gasnetex_obcount = existing_default_args->find("-gex:obcount") != std::string::npos;
#endif
    ss << std::regex_replace(
      *existing_default_args, std::regex{R"(\-ll:networks\s+\w+)", std::regex::optimize}, "");
  }

  if (need_network_init) {
    if constexpr (!LEGATE_DEFINED(LEGATE_USE_NETWORK)) {
      throw TracedException<std::runtime_error>{
        "Legate was run on multiple nodes but was not built with networking support. Please "
        "install Legate again with networking support (e.g. configured \"--with-ucx\")"};
    }

    // We have to pass an explicit `-ll:networks` flag, otherwise Realm will silently continue with
    // single-node execution if network initialization fails. Therefore, even though Realm's default
    // priority list for network modules is good enough for us, we have to duplicate it here.
#ifdef REALM_USE_UCX
    ss << " -ll:networks ucx";
#endif
#ifdef REALM_USE_GASNETEX
    ss << " -ll:networks gasnetex";

    if (!user_set_gasnetex_obcount) {
      // Need to set -gex:obcount appropriately, see
      // https://github.com/StanfordLegion/realm/issues/239
      const auto ranks         = static_cast<std::uint64_t>(num_ranks());
      const auto gpus_per_rank = gpus_per_rank_for_gasnetex_obcount();
      const auto obcount       = (4 + 2 * gpus_per_rank) * ranks;

      log_legate().debug() << "GEX obcount sizing: ranks=" << ranks
                           << " gpus_per_rank=" << gpus_per_rank << " obcount=" << obcount;

      ss << " -gex:obcount " << obcount;
    }
#endif
#ifdef REALM_USE_GASNET1
    ss << " -ll:networks gasnet1";
#endif
#ifdef REALM_USE_MPI
    ss << " -ll:networks mpi";
#endif
  } else {
    ss << " -ll:networks none";
  }

  const auto realm_default_args = ss.str();

  REALM_DEFAULT_ARGS.set(realm_default_args);
  log_legate().debug() << "REALM_DEFAULT_ARGS=" << realm_default_args;
}

void set_env_vars()
{
  if (LEGATE_DEFINED(LEGATE_USE_NCCL)) {
    // If using NCCL prefer parallel launch mode over cooperative groups, as the former plays
    // better with Realm.
    EnvironmentVariable<std::string>{"NCCL_LAUNCH_MODE"}.set("PARALLEL", /* overwrite */ true);
  }
#if defined(REALM_USE_GASNETEX) || defined(REALM_USE_GASNET1)
  // Make sure GASNet initializes MPI with the right level of threading support
  EnvironmentVariable<std::string>{"GASNET_MPI_THREAD"}.set("MPI_THREAD_MULTIPLE",
                                                            /* overwrite */ true);
#endif

  // Make sure we do not use the CUDA_TL in UCC CUDA_TL currently only works in a single-thread
  // setting. CUDA_TL could work with Legate multi-rank where every rank has a single thread,
  // but we just blanket disable it.
  EnvironmentVariable<std::string>{"UCC_TLS"}.set("^cuda", /* overwrite */ true);

#ifdef REALM_USE_UCX
  EnvironmentVariable<std::string>{"UCX_CUDA_COPY_MAX_REG_RATIO"}.set("1.0",
                                                                      /* overwrite */ true);
  EnvironmentVariable<std::string>{"UCX_RCACHE_PURGE_ON_FORK"}.set("n",
                                                                   /* overwrite */ true);
#endif

  if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    EnvironmentVariable<std::uint32_t>{"CUTENSOR_LOG_LEVEL"}.set(/*value=*/1,
                                                                 /* overwrite */ false);
  }

  EnvironmentVariable<bool>{"REALM_BACKTRACE"}.set(/*value=*/false, /* overwrite */ false);
}

}  // namespace

/*static*/ void Runtime::start()
{
  // Call as soon as possible, to ensure that any exceptions are pretty-printed
  static_cast<void>(install_terminate_handler());

  if (has_started()) {
    return;
  }

  if (has_finished()) {
    throw TracedException<std::runtime_error>{
      "Legate runtime cannot be started after legate::finish is called"};
  }

  set_env_vars();

  const auto need_network_init = multi_node_job() || REALM_UCP_BOOTSTRAP_MODE.get() == "p2p";
  // only do MPI version detection if we're running a multi-node job or using p2p network
  // bootstrap
  if (need_network_init) {
    set_mpi_wrapper_libraries();
  }

  handle_realm_default_args(need_network_init);

  int argc                           = 1;
  const char* const dummy_argv_arr[] = {"legate-placeholder-binary-name", nullptr};
  // Realm won't modify the existing strings, but nevertheless they require a char*
  char** dummy_argv = const_cast<char**>(dummy_argv_arr);
  char** argv       = dummy_argv;

  // If Realm finds anything in REALM_DEFAULT_ARGS, it will copy it onto the command line, right
  // after the (fake) program name. So at exit we should free everything except the first token.
  LEGATE_SCOPE_GUARD(
    if (argv != dummy_argv) {
      for (int i = 1; i < argc; ++i) {
        std::free(argv[i]);
      }
      std::free(static_cast<void*>(argv));
    });

  // Under control replication mode, all ranks return from Legion::Runtime::start(); however,
  // under single controller execution mode, only the controller rank returns immediately.
  // The worker ranks return when the Realm runtime is destructed by the controller.

  auto runtime = Realm::Runtime::get_runtime();

  auto&& check_realm = [&](auto result) {
    if (!result) {
      throw TracedException<std::runtime_error>{fmt::format("Realm Runtime failed to start")};
    }
  };

  check_realm(runtime.network_init(&argc, &argv));
  check_realm(runtime.create_configs(argc, argv));

  auto config = handle_legate_args();

  runtime.parse_command_line(argc, argv);
  runtime.finish_configure();

  runtime.start();

  initialize_(std::move(config));
}

namespace {

class RuntimeManager {
 public:
  enum class State : std::uint8_t { UNINITIALIZED, INITIALIZED, FINALIZING, FINALIZED };

  /**
   * @brief Construct the singleton runtime object.
   *
   * @param config The config parameter to construct the runtime with.
   *
   * @throw std::runtime_error If the runtime has already been constructed.
   */
  [[nodiscard]] Runtime& construct_runtime(Config config);

  [[nodiscard]] Runtime& get();
  [[nodiscard]] State state() const noexcept;
  void mark_finalizing() noexcept;
  void reset() noexcept;

 private:
  std::atomic<State> state_{State::UNINITIALIZED};
  std::optional<Runtime> rt_{};
};

Runtime& RuntimeManager::construct_runtime(Config config)
{
  if (state() != State::UNINITIALIZED) {
    throw TracedException<std::runtime_error>{
      "Legate runtime has already been constructed or finalized, and cannot be re-initialized "
      "without restarting the program."};
  }

  LEGATE_CHECK(!rt_.has_value());
  rt_.emplace(std::move(config));
  state_.store(State::INITIALIZED, std::memory_order_release);
  return *rt_;
}

Runtime& RuntimeManager::get()
{
  if (LEGATE_LIKELY(rt_.has_value())) {
    if (LEGATE_LIKELY(state() == State::INITIALIZED)) {
      return *rt_;
    }
    // The Runtime object still exists while FINALIZING, but Legion may already
    // be shutting down. Do not let late destructors call back into Legate here.
    throw TracedException<std::runtime_error>{
      "Legate runtime is finalizing, and cannot be retrieved safely."};
  }

  switch (state()) {
    case State::UNINITIALIZED:
      throw TracedException<std::runtime_error>{
        "Must call legate::start() before retrieving the Legate runtime."};
    case State::INITIALIZED:
      // This should never happen
      throw TracedException<std::runtime_error>{
        "Legate is in an inconsistent state. Legate claims to have initialized the runtime, but "
        "the runtime does not exist. Please report this bug to the Legate developers by opening "
        "an issue at https://github.com/nv-legate/legate/issues. This error is very likely "
        "unrecoverable; the user is advised to restart their program."};
    case State::FINALIZING:
    case State::FINALIZED:
      // Legion currently does not allow re-initialization after shutdown, so we need to track
      // this ourselves...
      throw TracedException<std::runtime_error>{
        "Legate runtime has been finalized, and cannot be re-initialized without restarting the "
        "program."};
  }

  LEGATE_ABORT("Unhandled runtime state: ", to_underlying(state()));
}

RuntimeManager::State RuntimeManager::state() const noexcept
{
  return state_.load(std::memory_order_acquire);
}

void RuntimeManager::mark_finalizing() noexcept
{
  state_.store(State::FINALIZING, std::memory_order_release);
}

void RuntimeManager::reset() noexcept
{
  rt_.reset();
  state_.store(State::FINALIZED, std::memory_order_release);
}

RuntimeManager the_runtime{};

}  // namespace

/*static*/ Runtime& Runtime::get_runtime() { return the_runtime.get(); }

void Runtime::register_shutdown_callback(ShutdownCallback callback)
{
  callbacks_.emplace_back(std::move(callback));
}

std::int32_t Runtime::finish()
{
  if (!has_started()) {
    return 0;
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Destroying Legate runtime...";
  }

  // NOLINTNEXTLINE(modernize-loop-convert) `callbacks_` can be modified by callbacks
  for (auto it = callbacks_.begin(); it != callbacks_.end(); ++it) {
    (*it)();
  }

  // Flush any outstanding operations before we tear down the runtime
  flush_scheduling_window();

  // Need a fence to make sure all client operations come before the subsequent clean-up tasks
  issue_execution_fence();

  // Destroy all communicators. This will likely launch some tasks for the clean-up.
  communicator_manager().destroy();

  // We're about to deallocate objects below, so let's block on all outstanding Legion operations
  issue_execution_fence(true);

  // Any STL containers holding Legion handles need to be cleared here, otherwise they cause
  // trouble when they get destroyed in the Legate runtime's destructor

  {
    comm::coll::BackendNetwork::get_network()->comm_destroy(cpu_communicator_.get());
    cpu_communicator_.reset();
  }

  if (config().need_cuda()) {
    // TODO(wonchanl): Error checking
    ncclCommDestroy(gpu_communicator_);
  }

  communicator_manager_.reset();
  partition_manager_.reset();
  scope_ = Scope{config()};
  comm::coll::finalize();
  // Mappers get raw pointers to Libraries, so just in case any of the above launched residual
  // cleanup tasks, we issue another fence here before we clear the Libraries.
  issue_execution_fence(true);

  if (config().need_cuda()) {
    auto&& api = cuda::detail::get_cuda_driver_api();

    api->stream_synchronize(current_stream_);
    api->stream_destroy(&current_stream_);
    api->device_primary_ctx_release(current_device_);
    current_device_ = -1;
  }

  libraries_.clear();
  core_library_.reset();
  // Reset this here and now (instead of waiting to destroy it when Runtime gets destroyed)
  // because the dtor of the module manager tries to get the runtime, which it should not do if
  // we are in the middle of self-destructing.
  cu_mod_manager_.reset();
  // This should be empty at this point, since the execution fence will ensure they are all
  // raised, but just in case, clear them. There is no hope of properly handling them now.
  initialized_ = false;

  // Legion may shut down as soon as wait_for_shutdown() observes that all
  // processes are done. Make the shutdown state visible to Realm worker
  // threads before that point so late Legate object destruction can avoid
  // calling back into Legion.
  the_runtime.mark_finalizing();
  // The previous call is asynchronous so we still need to
  // wait for the shutdown of the runtime to complete
  Realm::Runtime::get_runtime().shutdown();
  auto result = Realm::Runtime::get_runtime().wait_for_shutdown();
  // Do NOT delete, move, re-order, or otherwise modify the following lines under ANY
  // circumstances.
  //
  // They must stay exactly as they are. the_runtime.reset() calls "delete this", and hence any
  // modification of the runtime object, or any of its derivatives hereafter is strictly
  // undefined behavior.
  //
  // BEGIN DO NOT MODIFY
  the_runtime.reset();
  return result;
  // END DO NOT MODIFY
}

namespace {

void register_legate_core_tasks(Library& core_lib)
{
  comm::register_tasks(core_lib);
  legate::experimental::io::detail::register_tasks();
}

}  // namespace

/*static*/ void Runtime::initialize_(Config config)
{
  auto& runtime = the_runtime.construct_runtime(std::move(config));

  ResourceConfig lib_config;

  lib_config.max_tasks       = CoreTask::MAX_TASK;
  lib_config.max_dyn_tasks   = lib_config.max_tasks - CoreTask::FIRST_DYNAMIC_TASK;
  lib_config.max_projections = to_underlying(CoreProjectionOp::MAX_FUNCTOR);
  // We register one sharding functor for each new projection functor
  lib_config.max_shardings     = to_underlying(CoreShardID::MAX_FUNCTOR);
  lib_config.max_reduction_ops = to_underlying(CoreReductionOp::MAX_REDUCTION);

  auto&& core_lib = runtime.create_library(
    CORE_LIBRARY_NAME, lib_config, mapping::detail::create_core_mapper(), {});
  // Order is deliberate. core_library_() must be set here, because the core mapper and mapper
  // manager expect to call get_runtime()->core_library().
  runtime.core_library_ = core_lib;

  // These must occur here in case the task registration routines below need to access the CUDA
  // driver API
  if (runtime.config().need_cuda()) {
    runtime.cu_mod_manager_.emplace(cuda::detail::get_cuda_driver_api());

    auto&& api = cuda::detail::get_cuda_driver_api();

    auto count              = api->device_get_count();
    auto ordinal            = runtime.local_machine().node_id % count;
    runtime.current_device_ = api->device_get(ordinal);

    static_cast<void>(api->device_primary_ctx_retain(runtime.current_device_));

    log_legate().print() << "Rank " << runtime.local_machine().node_id << " gets GPU " << ordinal;

    {
      const cuda::detail::AutoPrimaryContext ctx{runtime.get_current_cuda_device()};

      runtime.current_stream_ = api->stream_create(0x01 /*CU_STREAM_NON_BLOCKING*/);
      api->mem_set_default_pool_release_threshold(ordinal, UINT64_MAX);
    }
  }
  comm::coll::init();
  runtime.communicator_manager_.emplace();
  runtime.partition_manager_.emplace();
  static_cast<void>(runtime.scope_.exchange_machine(runtime.create_toplevel_machine()));

  // TODO(wonchanl): the cpu communicator assumes rank-per-processor
  {
    auto&& network       = comm::coll::BackendNetwork::get_network();
    auto&& local_machine = runtime.local_machine();

    auto communicator = std::make_unique<legate::comm::coll::Coll_Comm>();

    std::vector<int> mapping_table;

    mapping_table.reserve(local_machine.total_nodes);
    for (std::uint32_t node_id = 0; node_id < local_machine.total_nodes; ++node_id) {
      mapping_table.push_back(node_id);
    }

    network->comm_create(communicator.get(),
                         local_machine.total_nodes,
                         local_machine.node_id,
                         network->init_comm(),
                         mapping_table.data());

    runtime.cpu_communicator_ = std::move(communicator);
  }

  if (runtime.config().need_cuda()) {
    auto&& network       = comm::coll::BackendNetwork::get_network();
    auto&& local_machine = runtime.local_machine();

    std::vector<ncclUniqueId> comm_ids{};

    comm_ids.resize(local_machine.total_nodes);

    if (local_machine.node_id == 0) {
      // TODO(wonchanl): Error checking
      ncclGetUniqueId(&comm_ids[0]);
    }

    network->all_gather(&comm_ids[local_machine.node_id],
                        comm_ids.data(),
                        sizeof(ncclUniqueId),
                        legate::comm::coll::CollDataType::CollInt8,
                        runtime.cpu_communicator_.get());

    const cuda::detail::AutoPrimaryContext ctx{runtime.get_current_cuda_device()};

    // TODO(wonchanl): Error checking
    ncclCommInitRank(
      &runtime.gpu_communicator_, local_machine.total_nodes, comm_ids[0], local_machine.node_id);
  }

  comm::register_builtin_communicator_factories(runtime.core_library());

  register_legate_core_tasks(core_lib);

  runtime.initialized_ = true;
}

CUstream Runtime::get_cuda_stream() const { return current_stream_; }

CUdevice Runtime::get_current_cuda_device() const { return current_device_; }

legate::comm::coll::CollComm Runtime::get_cpu_comm() const { return cpu_communicator_.get(); }

ncclComm_t Runtime::get_gpu_comm() const { return gpu_communicator_; }

cuda::detail::CUDAModuleManager& Runtime::get_cuda_module_manager()
{
  // This needs to be initialized in a thread-safe manner, so we do it in
  // Runtime::initialize(). Hence, we want this to throw if it is accessed before then, because
  // we can't guarantee it's threadsafe (and don't want to wrap this in some kind of mutex...).
  if (!cu_mod_manager_.has_value()) {
    throw TracedException<std::runtime_error>{
      "CUDA module manager is unavailable because Legate was started without GPUs"};
  }

  return cu_mod_manager_.value();
}

Processor Runtime::get_executing_processor() const
{
  auto& procs = local_machine().has_gpus() ? local_machine().gpus() : local_machine().cpus();

  return procs[node_id() % procs.size()];
}

bool has_started() { return the_runtime.state() == RuntimeManager::State::INITIALIZED; }

bool has_finished()
{
  const auto state = the_runtime.state();

  return state == RuntimeManager::State::FINALIZING || state == RuntimeManager::State::FINALIZED;
}

}  // namespace legate::detail
