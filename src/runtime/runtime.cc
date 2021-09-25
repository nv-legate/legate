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

#include "legate.h"
#include "mapping/core_mapper.h"
#include "runtime/context.h"
#include "runtime/projection.h"
#include "runtime/shard.h"
#include "utilities/deserializer.h"
#ifdef LEGATE_USE_CUDA
#include "gpu/cudalibs.h"
#endif

namespace legate {

using namespace Legion;

Logger log_legate("legate");

// This is the unique string name for our library which can be used
// from both C++ and Python to generate IDs
static const char* const core_library_name = "legate.core";

/*static*/ bool Core::show_progress = false;

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

static void toplevel_task(const Task* task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx,
                          Legion::Runtime* legion_runtime)
{
  auto runtime = Runtime::get_runtime();
  auto main    = runtime->get_main_function();

  auto args = Legion::Runtime::get_input_args();

  main(args.argc, args.argv, runtime);
}

static void initialize_cpu_resource_task(const Task* task,
                                         const std::vector<PhysicalRegion>& regions,
                                         Context ctx,
                                         Legion::Runtime* runtime)
{
  // Nothing to do here yet...
}

static void finalize_cpu_resource_task(const Task* task,
                                       const std::vector<PhysicalRegion>& regions,
                                       Context ctx,
                                       Legion::Runtime* runtime)
{
  // Nothing to do here yet...
}

static ReturnValues extract_scalar_task(const Task* task,
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
static void initialize_gpu_resource_task(const Task* task,
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

static void finalize_gpu_resource_task(const Task* task,
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
    auto registrar = make_registrar(toplevel_task_id, toplevel_task_name, Processor::LOC_PROC);
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

  auto runtime  = Runtime::get_runtime();
  auto core_lib = runtime->find_library(core_library_name);
  legion_runtime->set_top_level_task_id(core_lib->get_task_id(LEGATE_CORE_TOPLEVEL_TASK_ID));
  legion_runtime->set_top_level_task_mapper_id(core_lib->get_mapper_id(0));
}

/*static*/ Runtime* Runtime::runtime_;

Runtime::Runtime() {}

Runtime::~Runtime()
{
  for (auto& pair : libraries_) delete pair.second;
}

void Runtime::set_main_function(MainFnPtr main_fn) { main_fn_ = main_fn; }

Runtime::MainFnPtr Runtime::get_main_function() const { return main_fn_; }

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

/*static*/ void Runtime::initialize(int32_t argc, char** argv)
{
  Legion::Runtime::initialize(&argc, &argv, true /*filter legion and realm args*/);
  Legion::Runtime::perform_registration_callback(legate::core_library_bootstrapping_callback,
                                                 true /*global*/);
  runtime_ = new Runtime();
}

/*static*/ int32_t Runtime::start(int32_t argc, char** argv)
{
  return Legion::Runtime::start(argc, argv);
}

/*static*/ Runtime* Runtime::get_runtime() { return Runtime::runtime_; }

void initialize(int32_t argc, char** argv) { Runtime::initialize(argc, argv); }

void set_main_function(Runtime::MainFnPtr main_fn)
{
  Runtime::get_runtime()->set_main_function(main_fn);
}

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
