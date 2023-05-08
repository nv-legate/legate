/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <algorithm>
#include <cmath>

#include "core/comm/comm.h"
#include "core/data/logical_region_field.h"
#include "core/data/logical_store.h"
#include "core/data/logical_store_detail.h"
#include "core/mapping/core_mapper.h"
#include "core/mapping/default_mapper.h"
#include "core/mapping/machine.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/partitioner.h"
#include "core/runtime/context.h"
#include "core/runtime/launcher.h"
#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "core/task/exception.h"
#include "core/task/task.h"
#include "core/type/type_info.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/machine.h"
#include "core/utilities/nvtx_help.h"
#include "legate.h"

namespace legate {

Logger log_legate("legate");

// This is the unique string name for our library which can be used
// from both C++ and Python to generate IDs
static const char* const core_library_name = "legate.core";

/*static*/ bool Core::show_progress_requested = false;

/*static*/ bool Core::use_empty_task = false;

/*static*/ bool Core::synchronize_stream_view = false;

/*static*/ bool Core::log_mapping_decisions = false;

/*static*/ bool Core::has_socket_mem = false;

/*static*/ bool Core::standalone = false;

/*static*/ LegateMainFnPtr Core::main_fn = nullptr;

/*static*/ void Core::parse_config(void)
{
#ifndef LEGATE_USE_CUDA
  const char* need_cuda = getenv("LEGATE_NEED_CUDA");
  if (need_cuda != nullptr) {
    fprintf(stderr,
            "Legate was run with GPUs but was not built with GPU support. "
            "Please install Legate again with the \"--cuda\" flag.\n");
    exit(1);
  }
#endif
#ifndef LEGATE_USE_OPENMP
  const char* need_openmp = getenv("LEGATE_NEED_OPENMP");
  if (need_openmp != nullptr) {
    fprintf(stderr,
            "Legate was run with OpenMP processors but was not built with "
            "OpenMP support. Please install Legate again with the \"--openmp\" flag.\n");
    exit(1);
  }
#endif
#ifndef LEGATE_USE_NETWORK
  const char* need_network = getenv("LEGATE_NEED_NETWORK");
  if (need_network != nullptr) {
    fprintf(stderr,
            "Legate was run on multiple nodes but was not built with networking "
            "support. Please install Legate again with \"--network\".\n");
    exit(1);
  }
#endif
  auto parse_variable = [](const char* variable, bool& result) {
    const char* value = getenv(variable);
    if (value != nullptr && atoi(value) > 0) result = true;
  };

  parse_variable("LEGATE_SHOW_PROGRESS", show_progress_requested);
  parse_variable("LEGATE_EMPTY_TASK", use_empty_task);
  parse_variable("LEGATE_SYNC_STREAM_VIEW", synchronize_stream_view);
  parse_variable("LEGATE_LOG_MAPPING", log_mapping_decisions);
}

static void toplevel_task(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* legion_runtime)
{
  auto runtime = Runtime::get_runtime();
  runtime->post_startup_initialization(ctx);

  if (nullptr == Core::main_fn) {
    log_legate.error(
      "No main function was provided. Please register one with 'legate::set_main_function'.");
    LEGATE_ABORT;
  }

  auto args = Legion::Runtime::get_input_args();
  Core::main_fn(args.argc, args.argv);
}

static void extract_scalar_task(
  const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
{
  // Legion preamble
  const Legion::Task* task;
  const std::vector<Legion::PhysicalRegion>* regions;
  Legion::Context legion_context;
  Legion::Runtime* runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

  Core::show_progress(task, legion_context, runtime);

  TaskContext context(task, *regions, legion_context, runtime);
  auto idx            = context.scalars()[0].value<int32_t>();
  auto value_and_size = ReturnValues::extract(task->futures[0], idx);

  // Legion postamble
  value_and_size.finalize(legion_context);
}

/*static*/ void Core::shutdown(void)
{
  // Nothing to do here yet...
}

/*static*/ void Core::show_progress(const Legion::Task* task,
                                    Legion::Context ctx,
                                    Legion::Runtime* runtime)
{
  if (!Core::show_progress_requested) return;
  const auto exec_proc     = runtime->get_executing_processor(ctx);
  const auto proc_kind_str = (exec_proc.kind() == Processor::LOC_PROC)   ? "CPU"
                             : (exec_proc.kind() == Processor::TOC_PROC) ? "GPU"
                                                                         : "OpenMP";

  std::stringstream point_str;
  const auto& point = task->index_point;
  point_str << point[0];
  for (int32_t dim = 1; dim < point.dim; ++dim) point_str << "," << point[dim];

  log_legate.print("%s %s task [%s], pt = (%s), proc = " IDFMT,
                   task->get_task_name(),
                   proc_kind_str,
                   task->get_provenance_string().c_str(),
                   point_str.str().c_str(),
                   exec_proc.id);
}

/*static*/ void Core::report_unexpected_exception(const Legion::Task* task,
                                                  const legate::TaskException& e)
{
  log_legate.error(
    "Task %s threw an exception \"%s\", but the task did not declare any exception. "
    "Please specify a Python exception that you want this exception to be re-thrown with "
    "using 'throws_exception'.",
    task->get_task_name(),
    e.error_message().c_str());
  LEGATE_ABORT;
}

/*static*/ void Core::retrieve_tunable(Legion::Context legion_context,
                                       Legion::Runtime* legion_runtime,
                                       LibraryContext* context)
{
  auto fut = legion_runtime->select_tunable_value(
    legion_context, LEGATE_CORE_TUNABLE_HAS_SOCKET_MEM, context->get_mapper_id());
  Core::has_socket_mem = fut.get_result<bool>();
}

void register_legate_core_tasks(Legion::Machine machine,
                                Legion::Runtime* runtime,
                                const LibraryContext* context)
{
  auto toplevel_task_id          = context->get_task_id(LEGATE_CORE_TOPLEVEL_TASK_ID);
  const char* toplevel_task_name = "Legate Core Toplevel Task";
  runtime->attach_name(
    toplevel_task_id, toplevel_task_name, false /*mutable*/, true /*local only*/);

  auto extract_scalar_task_id          = context->get_task_id(LEGATE_CORE_EXTRACT_SCALAR_TASK_ID);
  const char* extract_scalar_task_name = "core::extract_scalar";
  runtime->attach_name(
    extract_scalar_task_id, extract_scalar_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    Legion::TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variant for both CPUs and GPUs
  {
    Legion::TaskVariantRegistrar registrar(toplevel_task_id, toplevel_task_name);
    registrar.add_constraint(Legion::ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(false);
    registrar.set_inner(false);
    registrar.set_replicable(true);
    registrar.global_registration = false;
    runtime->register_task_variant<toplevel_task>(registrar, LEGATE_CPU_VARIANT);
  }
  // Register the task variants
  auto register_extract_scalar = [&](auto proc_kind, auto variant_id) {
    auto registrar = make_registrar(extract_scalar_task_id, extract_scalar_task_name, proc_kind);
    Legion::CodeDescriptor desc(extract_scalar_task);
    runtime->register_task_variant(
      registrar, desc, nullptr, 0, LEGATE_MAX_SIZE_SCALAR_RETURN, variant_id);
  };
  register_extract_scalar(Processor::LOC_PROC, LEGATE_CPU_VARIANT);
#ifdef LEGATE_USE_CUDA
  register_extract_scalar(Processor::TOC_PROC, LEGATE_GPU_VARIANT);
#endif
#ifdef LEGATE_USE_OPENMP
  register_extract_scalar(Processor::OMP_PROC, LEGATE_OMP_VARIANT);
#endif
  comm::register_tasks(machine, runtime, context);
}

extern void register_exception_reduction_op(Legion::Runtime* runtime,
                                            const LibraryContext* context);

#define BUILTIN_REDOP_ID(OP, TYPE_CODE) \
  (LEGION_REDOP_BASE + (OP)*LEGION_TYPE_TOTAL + (static_cast<int32_t>(TYPE_CODE)))

#define RECORD(OP, TYPE_CODE) \
  PrimitiveType(TYPE_CODE).record_reduction_operator(OP, BUILTIN_REDOP_ID(OP, TYPE_CODE));

#define RECORD_INT(OP)           \
  RECORD(OP, Type::Code::BOOL)   \
  RECORD(OP, Type::Code::INT8)   \
  RECORD(OP, Type::Code::INT16)  \
  RECORD(OP, Type::Code::INT32)  \
  RECORD(OP, Type::Code::INT64)  \
  RECORD(OP, Type::Code::UINT8)  \
  RECORD(OP, Type::Code::UINT16) \
  RECORD(OP, Type::Code::UINT32) \
  RECORD(OP, Type::Code::UINT64)

#define RECORD_FLOAT(OP)          \
  RECORD(OP, Type::Code::FLOAT16) \
  RECORD(OP, Type::Code::FLOAT32) \
  RECORD(OP, Type::Code::FLOAT64)

#define RECORD_COMPLEX(OP) RECORD(OP, Type::Code::COMPLEX64)

#define RECORD_ALL(OP) \
  RECORD_INT(OP)       \
  RECORD_FLOAT(OP)     \
  RECORD_COMPLEX(OP)

void register_builtin_reduction_ops()
{
  RECORD_ALL(ADD_LT)
  RECORD(ADD_LT, Type::Code::COMPLEX128)
  RECORD_ALL(SUB_LT)
  RECORD_ALL(MUL_LT)
  RECORD_ALL(DIV_LT)

  RECORD_INT(MAX_LT)
  RECORD_FLOAT(MAX_LT)

  RECORD_INT(MIN_LT)
  RECORD_FLOAT(MIN_LT)

  RECORD_INT(OR_LT)
  RECORD_INT(AND_LT)
  RECORD_INT(XOR_LT)
}

/*static*/ void core_library_registration_callback(Legion::Machine machine,
                                                   Legion::Runtime* legion_runtime,
                                                   const std::set<Processor>& local_procs)
{
  Runtime::create_runtime(legion_runtime);

  ResourceConfig config;
  config.max_tasks       = LEGATE_CORE_NUM_TASK_IDS;
  config.max_projections = LEGATE_CORE_MAX_FUNCTOR_ID;
  // We register one sharding functor for each new projection functor
  config.max_shardings     = LEGATE_CORE_MAX_FUNCTOR_ID;
  config.max_reduction_ops = LEGATE_CORE_MAX_REDUCTION_OP_ID;

  auto runtime  = Runtime::get_runtime();
  auto core_lib = runtime->create_library(core_library_name, config);

  register_legate_core_tasks(machine, legion_runtime, core_lib);

  register_legate_core_mapper(machine, legion_runtime, core_lib);

  register_builtin_reduction_ops();

  register_exception_reduction_op(legion_runtime, core_lib);

  register_legate_core_projection_functors(legion_runtime, core_lib);

  register_legate_core_sharding_functors(legion_runtime, core_lib);

  if (!Core::standalone)
    Runtime::get_runtime()->post_startup_initialization(Legion::Runtime::get_context());
}

/*static*/ void core_library_bootstrapping_callback(Legion::Machine machine,
                                                    Legion::Runtime* legion_runtime,
                                                    const std::set<Processor>& local_procs)
{
  Core::standalone = true;

  core_library_registration_callback(machine, legion_runtime, local_procs);

  auto runtime = Runtime::get_runtime();

  auto core_lib = runtime->find_library(core_library_name);
  legion_runtime->set_top_level_task_id(core_lib->get_task_id(LEGATE_CORE_TOPLEVEL_TASK_ID));
  legion_runtime->set_top_level_task_mapper_id(core_lib->get_mapper_id());

  Core::parse_config();
}

////////////////////////////////////////////////////
// legate::RegionManager
////////////////////////////////////////////////////

class RegionManager {
 public:
  RegionManager(Runtime* runtime, const Domain& shape);

 private:
  Legion::LogicalRegion active_region() const;
  void create_region();

 public:
  bool has_space() const;
  std::pair<Legion::LogicalRegion, Legion::FieldID> allocate_field(size_t field_size);
  void import_region(const Legion::LogicalRegion& region);

 private:
  Runtime* runtime_;
  Domain shape_;
  std::vector<Legion::LogicalRegion> regions_{};
};

RegionManager::RegionManager(Runtime* runtime, const Domain& shape)
  : runtime_(runtime), shape_(shape)
{
}

Legion::LogicalRegion RegionManager::active_region() const { return regions_.back(); }

void RegionManager::create_region()
{
  auto is = runtime_->find_or_create_index_space(shape_);
  auto fs = runtime_->create_field_space();
  regions_.push_back(runtime_->create_region(is, fs));
}

bool RegionManager::has_space() const { return regions_.size() > 0; }

std::pair<Legion::LogicalRegion, Legion::FieldID> RegionManager::allocate_field(size_t field_size)
{
  if (!has_space()) create_region();
  auto lr  = active_region();
  auto fid = runtime_->allocate_field(lr.get_field_space(), field_size);
  return std::make_pair(lr, fid);
}

void RegionManager::import_region(const Legion::LogicalRegion& region)
{
  regions_.push_back(region);
}

////////////////////////////////////////////////////
// legate::FieldManager
////////////////////////////////////////////////////

class FieldManager {
 public:
  FieldManager(Runtime* runtime, const Domain& shape, uint32_t field_size);

 public:
  std::shared_ptr<LogicalRegionField> allocate_field();
  std::shared_ptr<LogicalRegionField> import_field(const Legion::LogicalRegion& region,
                                                   Legion::FieldID field_id);

 private:
  Runtime* runtime_;
  Domain shape_;
  uint32_t field_size_;

 private:
  using FreeField = std::pair<Legion::LogicalRegion, Legion::FieldID>;
  std::deque<FreeField> free_fields_;
};

FieldManager::FieldManager(Runtime* runtime, const Domain& shape, uint32_t field_size)
  : runtime_(runtime), shape_(shape), field_size_(field_size)
{
}

std::shared_ptr<LogicalRegionField> FieldManager::allocate_field()
{
  LogicalRegionField* rf = nullptr;
  if (!free_fields_.empty()) {
    auto field = free_fields_.front();
    log_legate.debug("Field %u recycled in field manager %p", field.second, this);
    free_fields_.pop_front();
    rf = new LogicalRegionField(field.first, field.second);
  } else {
    auto rgn_mgr = runtime_->find_or_create_region_manager(shape_);
    Legion::LogicalRegion lr;
    Legion::FieldID fid;
    std::tie(lr, fid) = rgn_mgr->allocate_field(field_size_);
    rf                = new LogicalRegionField(lr, fid);
    log_legate.debug("Field %u created in field manager %p", fid, this);
  }
  assert(rf != nullptr);
  return std::shared_ptr<LogicalRegionField>(rf, [this](auto* field) {
    log_legate.debug("Field %u freed in field manager %p", field->field_id(), this);
    this->free_fields_.push_back(FreeField(field->region(), field->field_id()));
    delete field;
  });
}

std::shared_ptr<LogicalRegionField> FieldManager::import_field(const Legion::LogicalRegion& region,
                                                               Legion::FieldID field_id)
{
  // Import the region only if the region manager is created fresh
  auto rgn_mgr = runtime_->find_or_create_region_manager(shape_);
  if (!rgn_mgr->has_space()) rgn_mgr->import_region(region);

  log_legate.debug("Field %u imported in field manager %p", field_id, this);

  auto* rf = new LogicalRegionField(region, field_id);
  return std::shared_ptr<LogicalRegionField>(rf, [this](auto* field) {
    log_legate.debug("Field %u freed in field manager %p", field->field_id(), this);
    this->free_fields_.push_back(FreeField(field->region(), field->field_id()));
    delete field;
  });
}

////////////////////////////////////////////////////
// legate::PartitionManager
////////////////////////////////////////////////////

PartitionManager::PartitionManager(Runtime* runtime, const LibraryContext* context)
{
  auto mapper_id = context->get_mapper_id();
  num_pieces_    = runtime->get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_NUM_PIECES);
  min_shard_volume_ =
    runtime->get_tunable<int64_t>(mapper_id, LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME);

  assert(num_pieces_ > 0);
  assert(min_shard_volume_ > 0);

  int32_t remaining_pieces = num_pieces_;
  auto push_factors        = [&](auto prime) {
    while (remaining_pieces % prime == 0) {
      piece_factors_.push_back(prime);
      remaining_pieces /= prime;
    }
  };

  push_factors(11);
  push_factors(7);
  push_factors(5);
  push_factors(3);
  push_factors(2);
}

Shape PartitionManager::compute_launch_shape(const Shape& shape)
{
  // Easy case if we only have one piece: no parallel launch space
  if (num_pieces_ == 1) return {};

  // If we only have one point then we never do parallel launches
  if (shape.all([](auto extent) { return 1 == extent; })) return {};

  // Prune out any dimensions that are 1
  std::vector<size_t> temp_shape{};
  std::vector<uint32_t> temp_dims{};
  int64_t volume = 1;
  for (uint32_t dim = 0; dim < shape.size(); ++dim) {
    auto extent = shape[dim];
    if (1 == extent) continue;
    temp_shape.push_back(extent);
    temp_dims.push_back(dim);
    volume *= extent;
  }

  // Figure out how many shards we can make with this array
  int64_t max_pieces = (volume + min_shard_volume_ - 1) / min_shard_volume_;
  assert(max_pieces > 0);
  // If we can only make one piece return that now
  if (1 == max_pieces) return {};

  // Otherwise we need to compute it ourselves
  // TODO: a better heuristic here.
  //       For now if we can make at least two pieces then we will make N pieces.
  max_pieces = num_pieces_;

  // First compute the N-th root of the number of pieces
  uint32_t ndim = temp_shape.size();
  assert(ndim > 0);
  std::vector<size_t> temp_result{};

  if (1 == ndim) {
    // Easy one dimensional case
    temp_result.push_back(std::min<size_t>(temp_shape.front(), static_cast<size_t>(max_pieces)));
  } else if (2 == ndim) {
    if (volume < max_pieces) {
      // TBD: Once the max_pieces heuristic is fixed, this should never happen
      temp_result.swap(temp_shape);
    } else {
      // Two dimensional so we can use square root to try and generate as square a pieces
      // as possible since most often we will be doing matrix operations with these
      auto nx   = temp_shape[0];
      auto ny   = temp_shape[1];
      auto swap = nx > ny;
      if (swap) std::swap(nx, ny);
      auto n = std::sqrt(static_cast<double>(max_pieces) * nx / ny);

      // Need to constraint n to be an integer with numpcs % n == 0
      // try rounding n both up and down
      auto n1 = std::max<int64_t>(1, static_cast<int64_t>(std::floor(n + 1e-12)));
      while (max_pieces % n1 != 0) --n1;
      auto n2 = std::max<int64_t>(1, static_cast<int64_t>(std::floor(n - 1e-12)));
      while (max_pieces % n2 != 0) ++n2;

      // pick whichever of n1 and n2 gives blocks closest to square
      // i.e. gives the shortest long side
      auto side1 = std::max(nx / n1, ny / (max_pieces / n1));
      auto side2 = std::max(nx / n2, ny / (max_pieces / n2));
      auto px    = static_cast<size_t>(side1 <= side2 ? n1 : n2);
      auto py    = static_cast<size_t>(max_pieces / px);

      // we need to trim launch space if it is larger than the
      // original shape in one of the dimensions (can happen in
      // testing)
      if (swap) {
        temp_result.push_back(std::min(py, temp_shape[0]));
        temp_result.push_back(std::min(px, temp_shape[1]));
      } else {
        temp_result.push_back(std::min(px, temp_shape[1]));
        temp_result.push_back(std::min(py, temp_shape[0]));
      }
    }
  } else {
    // For higher dimensions we care less about "square"-ness and more about evenly dividing
    // things, compute the prime factors for our number of pieces and then round-robin them
    // onto the shape, with the goal being to keep the last dimension >= 32 for good memory
    // performance on the GPU
    temp_result.resize(ndim);
    std::fill(temp_result.begin(), temp_result.end(), 1);
    size_t factor_prod = 1;
    for (auto factor : piece_factors_) {
      // Avoid exceeding the maximum number of pieces
      if (factor * factor_prod > max_pieces) break;

      factor_prod *= factor;

      std::vector<size_t> remaining;
      for (uint32_t idx = 0; idx < temp_shape.size(); ++idx)
        remaining.push_back((temp_shape[idx] + temp_result[idx] - 1) / temp_result[idx]);
      uint32_t big_dim = std::max_element(remaining.begin(), remaining.end()) - remaining.begin();
      if (big_dim < ndim - 1) {
        // Not the last dimension, so do it
        temp_result[big_dim] *= factor;
      } else {
        // Last dim so see if it still bigger than 32
        if (remaining[big_dim] / factor >= 32) {
          // go ahead and do it
          temp_result[big_dim] *= factor;
        } else {
          // Won't be see if we can do it with one of the other dimensions
          uint32_t next_big_dim =
            std::max_element(remaining.begin(), remaining.end() - 1) - remaining.begin();
          if (remaining[next_big_dim] / factor > 0)
            temp_result[next_big_dim] *= factor;
          else
            // Fine just do it on the last dimension
            temp_result[big_dim] *= factor;
        }
      }
    }
  }

  // Project back onto the original number of dimensions
  assert(temp_result.size() == ndim);
  std::vector<size_t> result(shape.size(), 1);
  for (uint32_t idx = 0; idx < ndim; ++idx) result[temp_dims[idx]] = temp_result[idx];

  return Shape(std::move(result));
}

Shape PartitionManager::compute_tile_shape(const Shape& extents, const Shape& launch_shape)
{
  assert(extents.size() == launch_shape.size());
  Shape tile_shape;
  for (uint32_t idx = 0; idx < extents.size(); ++idx) {
    auto x = extents[idx];
    auto y = launch_shape[idx];
    tile_shape.append_inplace((x + y - 1) / y);
  }
  return std::move(tile_shape);
}

Legion::IndexPartition PartitionManager::find_index_partition(const Legion::IndexSpace& index_space,
                                                              const Tiling& tiling) const
{
  auto finder = tiling_cache_.find(std::make_pair(index_space, tiling));
  if (finder != tiling_cache_.end())
    return finder->second;
  else
    return Legion::IndexPartition::NO_PART;
}

void PartitionManager::record_index_partition(const Legion::IndexSpace& index_space,
                                              const Tiling& tiling,
                                              const Legion::IndexPartition& index_partition)
{
  tiling_cache_[std::make_pair(index_space, tiling)] = index_partition;
}

////////////////////////////////////////////////////
// legate::Runtime
////////////////////////////////////////////////////

/*static*/ Runtime* Runtime::runtime_;

namespace {

constexpr uint32_t CUSTOM_TYPE_UID_BASE = 1000;

}  // namespace

Runtime::Runtime(Legion::Runtime* legion_runtime)
  : legion_runtime_(legion_runtime), next_type_uid_(CUSTOM_TYPE_UID_BASE)
{
}

Runtime::~Runtime()
{
  for (auto& [_, context] : libraries_) delete context;
}

LibraryContext* Runtime::find_library(const std::string& library_name,
                                      bool can_fail /*=false*/) const
{
  auto finder = libraries_.find(library_name);
  if (libraries_.end() == finder) {
    if (!can_fail) {
      log_legate.error("Library %s does not exist", library_name.c_str());
      LEGATE_ABORT;
    } else
      return nullptr;
  }
  return finder->second;
}

LibraryContext* Runtime::create_library(const std::string& library_name,
                                        const ResourceConfig& config,
                                        std::unique_ptr<mapping::Mapper> mapper)
{
  if (libraries_.find(library_name) != libraries_.end()) {
    log_legate.error("Library %s already exists", library_name.c_str());
    LEGATE_ABORT;
  }

  log_legate.debug("Library %s is created", library_name.c_str());
  if (nullptr == mapper) mapper = std::make_unique<mapping::DefaultMapper>();
  auto context             = new LibraryContext(library_name, config, std::move(mapper));
  libraries_[library_name] = context;
  return context;
}

uint32_t Runtime::get_type_uid() { return next_type_uid_++; }

void Runtime::record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id)
{
#ifdef DEBUG_LEGATE
  log_legate.debug("Record reduction op (type_uid: %d, op_kind: %d, legion_op_id: %d)",
                   type_uid,
                   op_kind,
                   legion_op_id);
#endif
  auto key    = std::make_pair(type_uid, op_kind);
  auto finder = reduction_ops_.find(key);
  if (finder != reduction_ops_.end()) {
    std::stringstream ss;
    ss << "Reduction op " << op_kind << " already exists for type " << type_uid;
    throw std::invalid_argument(std::move(ss).str());
  }
  reduction_ops_.emplace(std::make_pair(key, legion_op_id));
}

int32_t Runtime::find_reduction_operator(int32_t type_uid, int32_t op_kind) const
{
  auto key    = std::make_pair(type_uid, op_kind);
  auto finder = reduction_ops_.find(key);
  if (reduction_ops_.end() == finder) {
#ifdef DEBUG_LEGATE
    log_legate.debug("Can't find reduction op (type_uid: %d, op_kind: %d)", type_uid, op_kind);
#endif
    std::stringstream ss;
    ss << "Reduction op " << op_kind << " does not exist for type " << type_uid;
    throw std::invalid_argument(std::move(ss).str());
  }
#ifdef DEBUG_LEGATE
  log_legate.debug(
    "Found reduction op %d (type_uid: %d, op_kind: %d)", finder->second, type_uid, op_kind);
#endif
  return finder->second;
}

void Runtime::enter_callback() { in_callback_ = true; }

void Runtime::exit_callback() { in_callback_ = false; }

bool Runtime::is_in_callback() const { return in_callback_; }

void Runtime::post_startup_initialization(Legion::Context legion_context)
{
  legion_context_    = legion_context;
  core_context_      = find_library(core_library_name);
  partition_manager_ = new PartitionManager(this, core_context_);
  Core::retrieve_tunable(legion_context_, legion_runtime_, core_context_);
  initialize_toplevel_machine();
}

// This function should be moved to the library context
std::unique_ptr<AutoTask> Runtime::create_task(LibraryContext* library, int64_t task_id)
{
  auto task = new AutoTask(library, task_id, next_unique_id_++);
  return std::unique_ptr<AutoTask>(task);
}

std::unique_ptr<ManualTask> Runtime::create_task(LibraryContext* library,
                                                 int64_t task_id,
                                                 const Shape& launch_shape)
{
  auto task = new ManualTask(library, task_id, launch_shape, next_unique_id_++);
  return std::unique_ptr<ManualTask>(task);
}

void Runtime::submit(std::unique_ptr<Operation> op)
{
  operations_.push_back(std::move(op));
  if (operations_.size() >= window_size_) {
    std::vector<std::unique_ptr<Operation>> to_schedule;
    to_schedule.swap(operations_);
    schedule(std::move(to_schedule));
  }
}

void Runtime::schedule(std::vector<std::unique_ptr<Operation>> operations)
{
  std::vector<Operation*> op_pointers{};
  op_pointers.reserve(operations.size());
  for (auto& op : operations) op_pointers.push_back(op.get());

  Partitioner partitioner(std::move(op_pointers));
  auto strategy = partitioner.solve();

  for (auto& op : operations) op->launch(strategy.get());
}

LogicalStore Runtime::create_store(std::unique_ptr<Type> type, int32_t dim)
{
  auto storage = std::make_shared<detail::Storage>(dim, std::move(type));
  return LogicalStore(std::make_shared<detail::LogicalStore>(std::move(storage)));
}

LogicalStore Runtime::create_store(std::vector<size_t> extents,
                                   std::unique_ptr<Type> type,
                                   bool optimize_scalar /*=false*/)
{
  auto storage = std::make_shared<detail::Storage>(extents, std::move(type), optimize_scalar);
  return LogicalStore(std::make_shared<detail::LogicalStore>(std::move(storage)));
}

LogicalStore Runtime::create_store(const Scalar& scalar)
{
  Shape extents{1};
  auto future  = create_future(scalar.ptr(), scalar.size());
  auto storage = std::make_shared<detail::Storage>(extents, scalar.type().clone(), future);
  return LogicalStore(std::make_shared<detail::LogicalStore>(std::move(storage)));
}

uint64_t Runtime::get_unique_store_id() { return next_store_id_++; }

std::shared_ptr<LogicalRegionField> Runtime::create_region_field(const Shape& extents,
                                                                 uint32_t field_size)
{
  DomainPoint lo, hi;
  hi.dim = lo.dim = static_cast<int32_t>(extents.size());
  assert(lo.dim <= LEGION_MAX_DIM);
  for (int32_t dim = 0; dim < lo.dim; ++dim) lo[dim] = 0;
  for (int32_t dim = 0; dim < lo.dim; ++dim) hi[dim] = extents[dim] - 1;

  Domain shape(lo, hi);
  auto fld_mgr = runtime_->find_or_create_field_manager(shape, field_size);
  return fld_mgr->allocate_field();
}

std::shared_ptr<LogicalRegionField> Runtime::import_region_field(Legion::LogicalRegion region,
                                                                 Legion::FieldID field_id,
                                                                 uint32_t field_size)
{
  // TODO: This is a blocking operation. We should instead use index sapces as keys to field
  // managers
  auto shape   = legion_runtime_->get_index_space_domain(legion_context_, region.get_index_space());
  auto fld_mgr = runtime_->find_or_create_field_manager(shape, field_size);
  return fld_mgr->import_field(region, field_id);
}

RegionField Runtime::map_region_field(LibraryContext* context, const LogicalRegionField* rf)
{
  auto region   = rf->region();
  auto field_id = rf->field_id();

  Legion::PhysicalRegion pr;

  RegionFieldID key(region, field_id);
  auto finder = inline_mapped_.find(key);
  if (inline_mapped_.end() == finder) {
    Legion::RegionRequirement req(region, READ_WRITE, EXCLUSIVE, region);
    req.add_field(field_id);

    auto mapper_id = context->get_mapper_id();
    // TODO: We need to pass the metadata about logical store
    Legion::InlineLauncher launcher(req, mapper_id);
    pr = legion_runtime_->map_region(legion_context_, launcher);
    inline_mapped_.insert({key, pr});
  } else
    pr = finder->second;
  return RegionField(rf->dim(), pr, field_id);
}

void Runtime::unmap_physical_region(Legion::PhysicalRegion pr)
{
  legion_runtime_->unmap_region(legion_context_, pr);
}

RegionManager* Runtime::find_or_create_region_manager(const Domain& shape)
{
  auto finder = region_managers_.find(shape);
  if (finder != region_managers_.end())
    return finder->second;
  else {
    auto rgn_mgr            = new RegionManager(this, shape);
    region_managers_[shape] = rgn_mgr;
    return rgn_mgr;
  }
}

FieldManager* Runtime::find_or_create_field_manager(const Domain& shape, uint32_t field_size)
{
  auto key    = FieldManagerKey(shape, field_size);
  auto finder = field_managers_.find(key);
  if (finder != field_managers_.end())
    return finder->second;
  else {
    auto fld_mgr         = new FieldManager(this, shape, field_size);
    field_managers_[key] = fld_mgr;
    return fld_mgr;
  }
}

PartitionManager* Runtime::partition_manager() const { return partition_manager_; }

Legion::IndexSpace Runtime::find_or_create_index_space(const Domain& shape)
{
  assert(nullptr != legion_context_);
  auto finder = index_spaces_.find(shape);
  if (finder != index_spaces_.end())
    return finder->second;
  else {
    auto is              = legion_runtime_->create_index_space(legion_context_, shape);
    index_spaces_[shape] = is;
    return is;
  }
}

Legion::IndexPartition Runtime::create_restricted_partition(
  const Legion::IndexSpace& index_space,
  const Legion::IndexSpace& color_space,
  Legion::PartitionKind kind,
  const Legion::DomainTransform& transform,
  const Domain& extent)
{
  return legion_runtime_->create_partition_by_restriction(
    legion_context_, index_space, color_space, transform, extent, kind);
}

Legion::FieldSpace Runtime::create_field_space()
{
  assert(nullptr != legion_context_);
  return legion_runtime_->create_field_space(legion_context_);
}

Legion::LogicalRegion Runtime::create_region(const Legion::IndexSpace& index_space,
                                             const Legion::FieldSpace& field_space)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->create_logical_region(legion_context_, index_space, field_space);
}

Legion::LogicalPartition Runtime::create_logical_partition(
  const Legion::LogicalRegion& logical_region, const Legion::IndexPartition& index_partition)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->get_logical_partition(legion_context_, logical_region, index_partition);
}

Legion::Future Runtime::create_future(const void* data, size_t datalen) const
{
  return Legion::Future::from_untyped_pointer(data, datalen);
}

Legion::FieldID Runtime::allocate_field(const Legion::FieldSpace& field_space, size_t field_size)
{
  assert(nullptr != legion_context_);
  auto allocator = legion_runtime_->create_field_allocator(legion_context_, field_space);
  return allocator.allocate_field(field_size);
}

Domain Runtime::get_index_space_domain(const Legion::IndexSpace& index_space) const
{
  assert(nullptr != legion_context_);
  return legion_runtime_->get_index_space_domain(legion_context_, index_space);
}

Legion::Future Runtime::dispatch(Legion::TaskLauncher* launcher,
                                 std::vector<Legion::OutputRequirement>* output_requirements)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->execute_task(legion_context_, *launcher, output_requirements);
}

Legion::FutureMap Runtime::dispatch(Legion::IndexTaskLauncher* launcher,
                                    std::vector<Legion::OutputRequirement>* output_requirements)
{
  assert(nullptr != legion_context_);
  return legion_runtime_->execute_index_space(legion_context_, *launcher, output_requirements);
}

Legion::Future Runtime::extract_scalar(const Legion::Future& result, uint32_t idx) const
{
  // FIXME: One of two things should happen:
  //   1) the variant should be picked based on available processor kinds
  //   2) the variant should be picked by the core mapper
  TaskLauncher launcher(core_context_, LEGATE_CORE_EXTRACT_SCALAR_TASK_ID, LEGATE_CPU_VARIANT);
  launcher.add_future(result);
  launcher.add_scalar(Scalar(idx));
  return launcher.execute_single();
}

Legion::FutureMap Runtime::extract_scalar(const Legion::FutureMap& result,
                                          uint32_t idx,
                                          const Legion::Domain& launch_domain) const
{
  // FIXME: One of two things should happen:
  //   1) the variant should be picked based on available processor kinds
  //   2) the variant should be picked by the core mapper
  TaskLauncher launcher(core_context_, LEGATE_CORE_EXTRACT_SCALAR_TASK_ID, LEGATE_CPU_VARIANT);
  launcher.add_future_map(result);
  launcher.add_scalar(Scalar(idx));
  return launcher.execute(launch_domain);
}

Legion::Future Runtime::reduce_future_map(const Legion::FutureMap& future_map,
                                          int32_t reduction_op) const
{
  return legion_runtime_->reduce_future_map(legion_context_,
                                            future_map,
                                            reduction_op,
                                            false /*deterministic*/,
                                            core_context_->get_mapper_id());
}

void Runtime::issue_execution_fence(bool block /*=false*/)
{
  auto future = legion_runtime_->issue_execution_fence(legion_context_);
  if (block) future.wait();
}

void Runtime::initialize_toplevel_machine()
{
  auto mapper_id = core_context_->get_mapper_id();
  auto num_nodes = get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_NUM_NODES);

  auto num_gpus = get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_GPUS);
  auto num_omps = get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_OMPS);
  auto num_cpus = get_tunable<int32_t>(mapper_id, LEGATE_CORE_TUNABLE_TOTAL_CPUS);

  auto create_range = [&num_nodes](int32_t num_procs) {
    auto per_node_count = num_procs / num_nodes;
    return mapping::ProcessorRange(0, num_procs, per_node_count);
  };

  auto machine = new mapping::MachineDesc({{mapping::TaskTarget::GPU, create_range(num_gpus)},
                                           {mapping::TaskTarget::OMP, create_range(num_omps)},
                                           {mapping::TaskTarget::CPU, create_range(num_cpus)}});
  machine_.reset(machine);
}

const mapping::MachineDesc& Runtime::get_machine() const
{
#ifdef DEBUG_LEGATE
  assert(machine_ != nullptr);
#endif
  return *machine_;
}

Legion::ProjectionID Runtime::get_projection(int32_t src_ndim, const proj::SymbolicPoint& point)
{
#ifdef DEBUG_LEGATE
  log_legate.debug() << "Query projection {src_ndim: " << src_ndim << ", point: " << point << "}";
#endif

  if (is_identity(src_ndim, point)) {
#ifdef DEBUG_LEGATE
    log_legate.debug() << "Identity projection {src_ndim: " << src_ndim << ", point: " << point
                       << "}";
#endif
    return 0;
  }

  ProjectionDesc key(src_ndim, point);
  auto finder = registered_projections_.find(key);
  if (registered_projections_.end() != finder) return finder->second;

  auto proj_id = core_context_->get_projection_id(next_projection_id_++);

  auto ndim = point.size();
  std::vector<int32_t> dims;
  std::vector<int32_t> weights;
  std::vector<int32_t> offsets;
  for (auto& expr : point.data()) {
    dims.push_back(expr.dim());
    weights.push_back(expr.weight());
    offsets.push_back(expr.offset());
  }
  legate_register_affine_projection_functor(
    src_ndim, ndim, dims.data(), weights.data(), offsets.data(), proj_id);
  registered_projections_[key] = proj_id;

#ifdef DEBUG_LEGATE
  log_legate.debug() << "Register projection " << proj_id << " {src_ndim: " << src_ndim
                     << ", point: " << point << "}";
#endif

  return proj_id;
}

Legion::ProjectionID Runtime::get_delinearizing_projection()
{
  return core_context_->get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID);
}

/*static*/ void Runtime::initialize(int32_t argc, char** argv)
{
  Legion::Runtime::initialize(&argc, &argv, true /*filter legion and realm args*/);
  Legion::Runtime::add_registration_callback(legate::core_library_bootstrapping_callback);
}

/*static*/ int32_t Runtime::start(int32_t argc, char** argv)
{
  return Legion::Runtime::start(argc, argv);
}

/*static*/ Runtime* Runtime::get_runtime() { return Runtime::runtime_; }

/*static*/ void Runtime::create_runtime(Legion::Runtime* legion_runtime)
{
  runtime_ = new Runtime(legion_runtime);
}

void initialize(int32_t argc, char** argv) { Runtime::initialize(argc, argv); }

void set_main_function(LegateMainFnPtr main_fn) { Core::main_fn = main_fn; }

int32_t start(int32_t argc, char** argv) { return Runtime::start(argc, argv); }

}  // namespace legate

extern "C" {

void legate_core_perform_registration()
{
  // Tell the runtime about our registration callback so we can register ourselves
  // Make sure it is global so this shared object always gets loaded on all nodes
  Legion::Runtime::perform_registration_callback(legate::core_library_registration_callback,
                                                 true /*global*/);
}
}
