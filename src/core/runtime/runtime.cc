/* Copyright 2021 NVIDIA Corporation
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

#include "core/data/logical_store.h"
#include "core/mapping/core_mapper.h"
#include "core/partitioning/partition.h"
#include "core/partitioning/partitioner.h"
#include "core/runtime/context.h"
#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "core/utilities/deserializer.h"
#include "legate.h"
#ifdef LEGATE_USE_CUDA
#include "core/gpu/cudalibs.h"
#endif

namespace legate {

using namespace Legion;

Logger log_legate("legate");

// This is the unique string name for our library which can be used
// from both C++ and Python to generate IDs
static const char* const core_library_name = "legate.core";

/*static*/ bool Core::show_progress = false;

/*static*/ LegateMainFnPtr Core::main_fn = nullptr;

/*static*/ void Core::parse_config(void)
{
#ifndef LEGATE_USE_CUDA
  const char* need_cuda = getenv("LEGATE_NEED_CUDA");
  if (need_cuda != NULL) {
    fprintf(stderr,
            "Legate was run with GPUs but was not built with GPU support. "
            "Please install Legate again with the \"--cuda\" flag.\n");
    exit(1);
  }
#endif
#ifndef LEGATE_USE_OPENMP
  const char* need_openmp = getenv("LEGATE_NEED_OPENMP");
  if (need_openmp != NULL) {
    fprintf(stderr,
            "Legate was run with OpenMP processors but was not built with "
            "OpenMP support. Please install Legate again with the \"--openmp\" flag.\n");
    exit(1);
  }
#endif
#ifndef LEGATE_USE_GASNET
  const char* need_gasnet = getenv("LEGATE_NEED_GASNET");
  if (need_gasnet != NULL) {
    fprintf(stderr,
            "Legate was run on multiple nodes but was not built with "
            "GASNet support. Please install Legate again with the \"--gasnet\" flag.\n");
    exit(1);
  }
#endif
  const char* progress = getenv("LEGATE_SHOW_PROGRESS");
  if (progress != NULL) show_progress = true;
}

#ifdef LEGATE_USE_CUDA
static CUDALibraries& get_cuda_libraries(Processor proc, bool check)
{
  if (proc.kind() != Processor::TOC_PROC) {
    fprintf(stderr, "Illegal request for CUDA libraries for non-GPU processor");
    LEGATE_ABORT
  }
  static std::map<Processor, CUDALibraries> cuda_libraries;
  std::map<Processor, CUDALibraries>::iterator finder = cuda_libraries.find(proc);
  if (finder == cuda_libraries.end()) {
    assert(!check);
    return cuda_libraries[proc];
  } else
    return finder->second;
}

/*static*/ cublasContext* Core::get_cublas(void)
{
  const Processor executing_processor = Processor::get_executing_processor();
  CUDALibraries& lib                  = get_cuda_libraries(executing_processor, true /*check*/);
  return lib.get_cublas();
}
#endif

static void toplevel_task(const Legion::Task* task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx,
                          Legion::Runtime* legion_runtime)
{
  auto runtime = Runtime::get_runtime();
  runtime->post_startup_initialization(ctx);

  if (nullptr == Core::main_fn) {
    log_legate.error(
      "No main function was provided. Please register one with 'legate::set_main_function'.");
    LEGATE_ABORT
  }

  auto args = Legion::Runtime::get_input_args();
  Core::main_fn(args.argc, args.argv);
}

static void initialize_cpu_resource_task(const Legion::Task* task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx,
                                         Legion::Runtime* runtime)
{
  // Nothing to do here yet...
}

static void finalize_cpu_resource_task(const Legion::Task* task,
                                       const std::vector<PhysicalRegion>& regions,
                                       Context ctx,
                                       Legion::Runtime* runtime)
{
  // Nothing to do here yet...
}

static ReturnValues extract_scalar_task(const Legion::Task* task,
                                        const std::vector<PhysicalRegion>& regions,
                                        Context legion_context,
                                        Legion::Runtime* runtime)
{
  TaskContext context(task, regions, legion_context, runtime);
  auto values = task->futures[0].get_result<ReturnValues>();
  auto idx    = context.scalars()[0].value<int32_t>();
  return ReturnValues({values[idx]});
}

#ifdef LEGATE_USE_CUDA
static void initialize_gpu_resource_task(const Legion::Task* task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx,
                                         Legion::Runtime* runtime)
{
  const LegateResource resource = *((const LegateResource*)task->args);
  switch (resource) {
    case LEGATE_CORE_RESOURCE_CUBLAS: {
      // This call will initialize cublas
      Core::get_cublas();
      break;
    }
    // TODO: implement support for other libraries
    case LEGATE_CORE_RESOURCE_CUDNN:
    case LEGATE_CORE_RESOURCE_CUDF:
    case LEGATE_CORE_RESOURCE_CUML:
    default: LEGATE_ABORT
  }
}

static void finalize_gpu_resource_task(const Legion::Task* task,
                                       const std::vector<PhysicalRegion>& regions,
                                       Context ctx,
                                       Legion::Runtime* runtime)
{
  CUDALibraries& libs = get_cuda_libraries(task->current_proc, true /*check*/);
  libs.finalize();
}
#endif  // LEGATE_USE_CUDA

/*static*/ void Core::shutdown(void)
{
  // Nothing to do here yet...
}

void register_legate_core_tasks(Machine machine,
                                Legion::Runtime* runtime,
                                const LibraryContext& context)
{
  const TaskID toplevel_task_id  = context.get_task_id(LEGATE_CORE_TOPLEVEL_TASK_ID);
  const char* toplevel_task_name = "Legate Core Toplevel Task";
  runtime->attach_name(
    toplevel_task_id, toplevel_task_name, false /*mutable*/, true /*local only*/);

  const TaskID initialize_task_id  = context.get_task_id(LEGATE_CORE_INITIALIZE_TASK_ID);
  const char* initialize_task_name = "Legate Core Resource Initialization";
  runtime->attach_name(
    initialize_task_id, initialize_task_name, false /*mutable*/, true /*local only*/);

  const TaskID finalize_task_id  = context.get_task_id(LEGATE_CORE_FINALIZE_TASK_ID);
  const char* finalize_task_name = "Legate Core Resource Finalization";
  runtime->attach_name(
    finalize_task_id, finalize_task_name, false /*mutable*/, true /*local only*/);

  const TaskID extract_scalar_task_id  = context.get_task_id(LEGATE_CORE_EXTRACT_SCALAR_TASK_ID);
  const char* extract_scalar_task_name = "Legate Core Scalar Extraction";
  runtime->attach_name(
    extract_scalar_task_id, extract_scalar_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(ProcessorConstraint(proc_kind));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variant for both CPUs and GPUs
  {
    TaskVariantRegistrar registrar(toplevel_task_id, toplevel_task_name);
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(false);
    registrar.set_inner(false);
    registrar.set_replicable(true);
    registrar.global_registration = false;
    runtime->register_task_variant<toplevel_task>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    auto registrar = make_registrar(initialize_task_id, initialize_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<initialize_cpu_resource_task>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    auto registrar = make_registrar(finalize_task_id, finalize_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<finalize_cpu_resource_task>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(extract_scalar_task_id, extract_scalar_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<ReturnValues, extract_scalar_task>(registrar,
                                                                      LEGATE_CPU_VARIANT);
  }
#ifdef LEGATE_USE_CUDA
  {
    auto registrar = make_registrar(initialize_task_id, initialize_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<initialize_gpu_resource_task>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    auto registrar = make_registrar(finalize_task_id, finalize_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<finalize_gpu_resource_task>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    // Make sure we fill in all the cuda libraries entries for the
    // local processors so we don't have races later
    Machine::ProcessorQuery local_gpus(machine);
    local_gpus.local_address_space();
    local_gpus.only_kind(Processor::TOC_PROC);
    for (auto local_gpu : local_gpus)
      // This call will make an entry for the CUDA libraries but not
      // initialize any of them
      get_cuda_libraries(local_gpu, false /*check*/);
  }
#endif
}

/*static*/ void core_library_registration_callback(Machine machine,
                                                   Legion::Runtime* legion_runtime,
                                                   const std::set<Processor>& local_procs)
{
  Runtime::create_runtime(legion_runtime);

  ResourceConfig config;
  config.max_tasks       = LEGATE_CORE_NUM_TASK_IDS;
  config.max_projections = LEGATE_CORE_MAX_FUNCTOR_ID;
  // We register one sharding functor for each new projection functor
  config.max_shardings = LEGATE_CORE_MAX_FUNCTOR_ID;

  auto runtime  = Runtime::get_runtime();
  auto core_lib = runtime->create_library(core_library_name, config);

  register_legate_core_tasks(machine, legion_runtime, *core_lib);

  register_legate_core_mapper(machine, legion_runtime, *core_lib);

  register_legate_core_projection_functors(legion_runtime, *core_lib);

  register_legate_core_sharding_functors(legion_runtime, *core_lib);
}

/*static*/ void core_library_bootstrapping_callback(Machine machine,
                                                    Legion::Runtime* legion_runtime,
                                                    const std::set<Processor>& local_procs)
{
  core_library_registration_callback(machine, legion_runtime, local_procs);

  auto runtime = Runtime::get_runtime();

  auto core_lib = runtime->find_library(core_library_name);
  legion_runtime->set_top_level_task_id(core_lib->get_task_id(LEGATE_CORE_TOPLEVEL_TASK_ID));
  legion_runtime->set_top_level_task_mapper_id(core_lib->get_mapper_id(0));

  Core::parse_config();
}

////////////////////////////////////////////////////
// legate::RegionManager
////////////////////////////////////////////////////

class RegionManager {
 public:
  RegionManager(Runtime* runtime, const Legion::Domain& shape);

 private:
  Legion::LogicalRegion active_region() const;
  bool has_space() const;
  void create_region();

 public:
  std::pair<Legion::LogicalRegion, Legion::FieldID> allocate_field(size_t field_size);

 private:
  Runtime* runtime_;
  Legion::Domain shape_;
  std::vector<Legion::LogicalRegion> regions_{};
};

RegionManager::RegionManager(Runtime* runtime, const Domain& shape)
  : runtime_(runtime), shape_(shape)
{
}

LogicalRegion RegionManager::active_region() const { return regions_.back(); }

bool RegionManager::has_space() const { return regions_.size() > 0; }

void RegionManager::create_region()
{
  auto is = runtime_->find_or_create_index_space(shape_);
  auto fs = runtime_->create_field_space();
  regions_.push_back(runtime_->create_region(is, fs));
}

std::pair<Legion::LogicalRegion, Legion::FieldID> RegionManager::allocate_field(size_t field_size)
{
  if (!has_space()) create_region();
  auto lr  = active_region();
  auto fid = runtime_->allocate_field(lr.get_field_space(), field_size);
  return std::make_pair(lr, fid);
}

////////////////////////////////////////////////////
// legate::FieldManager
////////////////////////////////////////////////////

class FieldManager {
 public:
  FieldManager(Runtime* runtime, const Legion::Domain& shape, LegateTypeCode code);

 public:
  std::shared_ptr<LogicalRegionField> allocate_field();

 private:
  Runtime* runtime_;
  Legion::Domain shape_;
  LegateTypeCode code_;
  size_t field_size_;
};

struct field_size_fn {
  template <LegateTypeCode CODE>
  size_t operator()()
  {
    return sizeof(legate_type_of<CODE>);
  }
};

static size_t get_field_size(LegateTypeCode code) { return type_dispatch(code, field_size_fn{}); }

FieldManager::FieldManager(Runtime* runtime, const Legion::Domain& shape, LegateTypeCode code)
  : runtime_(runtime), shape_(shape), code_(code), field_size_(get_field_size(code))
{
}

std::shared_ptr<LogicalRegionField> FieldManager::allocate_field()
{
  auto rgn_mgr = runtime_->find_or_create_region_manager(shape_);
  LogicalRegion lr;
  FieldID fid;
  std::tie(lr, fid) = rgn_mgr->allocate_field(field_size_);
  return std::make_shared<LogicalRegionField>(runtime_, lr, fid);
}

////////////////////////////////////////////////////
// legate::PartitionManager
////////////////////////////////////////////////////

PartitionManager::PartitionManager(Runtime* runtime, const LibraryContext* context)
{
  num_pieces_       = runtime->get_tunable<int32_t>(context, LEGATE_CORE_TUNABLE_NUM_PIECES);
  min_shard_volume_ = runtime->get_tunable<int64_t>(context, LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME);

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

std::vector<size_t> PartitionManager::compute_launch_shape(const std::vector<size_t>& shape)
{
  // Easy case if we only have one piece: no parallel launch space
  if (num_pieces_ == 1) return {};

  // If we only have one point then we never do parallel launches
  if (std::all_of(shape.begin(), shape.end(), [](auto extent) { return 1 == extent; })) return {};

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

  return std::move(result);
}

std::vector<size_t> PartitionManager::compute_tile_shape(const std::vector<size_t>& extents,
                                                         const std::vector<size_t>& launch_shape)
{
  assert(extents.size() == launch_shape.size());
  std::vector<size_t> tile_shape;
  for (uint32_t idx = 0; idx < extents.size(); ++idx) {
    auto x = extents[idx];
    auto y = launch_shape[idx];
    tile_shape.push_back((x + y - 1) / y);
  }
  return std::move(tile_shape);
}

////////////////////////////////////////////////////
// legate::Runtime
////////////////////////////////////////////////////

/*static*/ Runtime* Runtime::runtime_;

Runtime::Runtime(Legion::Runtime* legion_runtime) : legion_runtime_(legion_runtime) {}

Runtime::~Runtime()
{
  for (auto& pair : libraries_) delete pair.second;
}

LibraryContext* Runtime::find_library(const std::string& library_name,
                                      bool can_fail /*=false*/) const
{
  auto finder = libraries_.find(library_name);
  if (libraries_.end() == finder) {
    if (!can_fail) {
      log_legate.error("Library %s does not exist", library_name.c_str());
      LEGATE_ABORT
    } else
      return nullptr;
  }
  return finder->second;
}

LibraryContext* Runtime::create_library(const std::string& library_name,
                                        const ResourceConfig& config)
{
  if (libraries_.find(library_name) != libraries_.end()) {
    log_legate.error("Library %s already exists", library_name.c_str());
    LEGATE_ABORT
  }

  log_legate.debug("Library %s is created", library_name.c_str());
  auto context = new LibraryContext(Legion::Runtime::get_runtime(), library_name, config);
  libraries_[library_name] = context;
  return context;
}

void Runtime::post_startup_initialization(Legion::Context legion_context)
{
  legion_context_    = legion_context;
  core_context_      = find_library(core_library_name);
  partition_manager_ = new PartitionManager(this, core_context_);
}

// This function should be moved to the library context
std::unique_ptr<Task> Runtime::create_task(LibraryContext* library,
                                           int64_t task_id,
                                           int64_t mapper_id /*=0*/)
{
  return std::make_unique<Task>(this, library, task_id, next_unique_id_++, mapper_id);
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

  Partitioner partitioner(this, std::move(op_pointers));
  auto strategy = partitioner.solve();

  for (auto& op : operations) op->launch(strategy.get());
}

LogicalStore Runtime::create_store(std::vector<size_t> extents, LegateTypeCode code)
{
  return LogicalStore(this, code, extents);
}

LogicalStore Runtime::create_store(const Scalar& scalar)
{
  return LogicalStore(this, scalar.code(), scalar.ptr());
}

std::shared_ptr<LogicalRegionField> Runtime::create_region_field(const std::vector<size_t>& extents,
                                                                 LegateTypeCode code)
{
  DomainPoint lo, hi;
  hi.dim = lo.dim = static_cast<int32_t>(extents.size());
  assert(lo.dim <= LEGION_MAX_DIM);
  for (int32_t dim = 0; dim < lo.dim; ++dim) lo[dim] = 0;
  for (int32_t dim = 0; dim < lo.dim; ++dim) hi[dim] = extents[dim] - 1;

  Domain shape(lo, hi);
  auto fld_mgr = runtime_->find_or_create_field_manager(shape, code);
  return fld_mgr->allocate_field();
}

RegionField Runtime::map_region_field(LibraryContext* context,
                                      std::shared_ptr<LogicalRegionField> rf)
{
  auto region   = rf->region();
  auto field_id = rf->field_id();

  RegionRequirement req(region, READ_WRITE, EXCLUSIVE, region);
  req.add_field(field_id);

  auto mapper_id = context->get_mapper_id(0);
  // TODO: We need to pass the metadata about logical store
  InlineLauncher launcher(req, mapper_id);
  auto pr = legion_runtime_->map_region(legion_context_, launcher);
  return RegionField(rf->dim(), pr, field_id);
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

FieldManager* Runtime::find_or_create_field_manager(const Domain& shape, LegateTypeCode code)
{
  auto key    = std::make_pair(shape, code);
  auto finder = field_managers_.find(key);
  if (finder != field_managers_.end())
    return finder->second;
  else {
    auto fld_mgr         = new FieldManager(this, shape, code);
    field_managers_[key] = fld_mgr;
    return fld_mgr;
  }
}

PartitionManager* Runtime::get_partition_manager() { return partition_manager_; }

IndexSpace Runtime::find_or_create_index_space(const Domain& shape)
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

Legion::IndexPartition Runtime::create_index_partition(const Legion::IndexSpace& index_space,
                                                       const Legion::IndexSpace& color_space,
                                                       Legion::PartitionKind kind,
                                                       const PartitioningFunctor* functor)
{
  return functor->construct(legion_runtime_, legion_context_, index_space, color_space, kind);
}

FieldSpace Runtime::create_field_space()
{
  assert(nullptr != legion_context_);
  return legion_runtime_->create_field_space(legion_context_);
}

LogicalRegion Runtime::create_region(const IndexSpace& index_space, const FieldSpace& field_space)
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
  return Legion::Future::from_untyped_pointer(legion_runtime_, data, datalen);
}

FieldID Runtime::allocate_field(const FieldSpace& field_space, size_t field_size)
{
  assert(nullptr != legion_context_);
  auto allocator = legion_runtime_->create_field_allocator(legion_context_, field_space);
  return allocator.allocate_field(field_size);
}

Domain Runtime::get_index_space_domain(const IndexSpace& index_space) const
{
  assert(nullptr != legion_context_);
  return legion_runtime_->get_index_space_domain(legion_context_, index_space);
}

std::shared_ptr<LogicalStore> Runtime::dispatch(TaskLauncher* launcher)
{
  assert(nullptr != legion_context_);
  legion_runtime_->execute_task(legion_context_, *launcher);
  return nullptr;
}

std::shared_ptr<LogicalStore> Runtime::dispatch(IndexTaskLauncher* launcher)
{
  assert(nullptr != legion_context_);
  legion_runtime_->execute_index_space(legion_context_, *launcher);
  return nullptr;
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
