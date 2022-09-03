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

#include "core/comm/comm.h"
#include "core/mapping/core_mapper.h"
#include "core/runtime/context.h"
#include "core/runtime/projection.h"
#include "core/runtime/shard.h"
#include "core/task/exception.h"
#include "core/task/task.h"
#include "core/utilities/deserializer.h"
#include "legate.h"

namespace legate {

using namespace Legion;

Logger log_legate("legate");

// This is the unique string name for our library which can be used
// from both C++ and Python to generate IDs
static const char* const core_library_name = "legate.core";

/*static*/ bool Core::show_progress_requested = false;

/*static*/ bool Core::use_empty_task = false;

/*static*/ bool Core::synchronize_stream_view = false;

/*static*/ bool Core::log_mapping_decisions = false;

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
#ifndef LEGATE_USE_GASNET
  const char* need_gasnet = getenv("LEGATE_NEED_GASNET");
  if (need_gasnet != nullptr) {
    fprintf(stderr,
            "Legate was run on multiple nodes but was not built with "
            "GASNet support. Please install Legate again with the \"--gasnet\" flag.\n");
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

static void extract_scalar_task(
  const void* args, size_t arglen, const void* userdata, size_t userlen, Legion::Processor p)
{
  // Legion preamble
  const Legion::Task* task;
  const std::vector<Legion::PhysicalRegion>* regions;
  Legion::Context legion_context;
  Legion::Runtime* runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

  Core::show_progress(task, legion_context, runtime, task->get_task_name());

  TaskContext context(task, *regions, legion_context, runtime);
  auto values = task->futures[0].get_result<ReturnValues>();
  auto idx    = context.scalars()[0].value<int32_t>();

  // Legion postamble
  ReturnValues({values[idx]}).finalize(legion_context);
}

/*static*/ void Core::shutdown(void)
{
  // Nothing to do here yet...
}

/*static*/ void Core::show_progress(const Legion::Task* task,
                                    Legion::Context ctx,
                                    Legion::Runtime* runtime,
                                    const char* task_name)
{
  if (!Core::show_progress_requested) return;
  const auto exec_proc     = runtime->get_executing_processor(ctx);
  const auto proc_kind_str = (exec_proc.kind() == Legion::Processor::LOC_PROC)   ? "CPU"
                             : (exec_proc.kind() == Legion::Processor::TOC_PROC) ? "GPU"
                                                                                 : "OpenMP";

  std::stringstream point_str;
  const auto& point = task->index_point;
  point_str << point[0];
  for (int32_t dim = 1; dim < task->index_point.dim; ++dim) point_str << "," << point[dim];

  log_legate.print("%s %s task, pt = (%s), proc = " IDFMT,
                   task_name,
                   proc_kind_str,
                   point_str.str().c_str(),
                   exec_proc.id);
}

/*static*/ void Core::report_unexpected_exception(const char* task_name,
                                                  const legate::TaskException& e)
{
  log_legate.error(
    "Task %s threw an exception \"%s\", but the task did not declare any exception. "
    "Please specify a Python exception that you want this exception to be re-thrown with "
    "using 'throws_exception'.",
    task_name,
    e.error_message().c_str());
  LEGATE_ABORT;
}

void register_legate_core_tasks(Machine machine, Runtime* runtime, const LibraryContext& context)
{
  const TaskID extract_scalar_task_id  = context.get_task_id(LEGATE_CORE_EXTRACT_SCALAR_TASK_ID);
  const char* extract_scalar_task_name = "core::extract_scalar";
  runtime->attach_name(
    extract_scalar_task_id, extract_scalar_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(ProcessorConstraint(proc_kind));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variants
  {
    auto registrar =
      make_registrar(extract_scalar_task_id, extract_scalar_task_name, Processor::LOC_PROC);
    Legion::CodeDescriptor desc(extract_scalar_task);
    runtime->register_task_variant(
      registrar, desc, nullptr, 0, LEGATE_MAX_SIZE_SCALAR_RETURN, LEGATE_CPU_VARIANT);
  }
  comm::register_tasks(machine, runtime, context);
}

extern void register_exception_reduction_op(Runtime* runtime, const LibraryContext& context);

/*static*/ void core_registration_callback(Machine machine,
                                           Runtime* runtime,
                                           const std::set<Processor>& local_procs)
{
  ResourceConfig config;
  config.max_tasks       = LEGATE_CORE_NUM_TASK_IDS;
  config.max_projections = LEGATE_CORE_MAX_FUNCTOR_ID;
  // We register one sharding functor for each new projection functor
  config.max_shardings     = LEGATE_CORE_MAX_FUNCTOR_ID;
  config.max_reduction_ops = LEGATE_CORE_MAX_REDUCTION_OP_ID;
  LibraryContext context(runtime, core_library_name, config);

  register_legate_core_tasks(machine, runtime, context);

  register_legate_core_mapper(machine, runtime, context);

  register_exception_reduction_op(runtime, context);

  register_legate_core_projection_functors(runtime, context);

  register_legate_core_sharding_functors(runtime, context);
}

}  // namespace legate

extern "C" {

void legate_core_perform_registration()
{
  // Tell the runtime about our registration callback so we can register ourselves
  // Make sure it is global so this shared object always gets loaded on all nodes
  Legion::Runtime::perform_registration_callback(legate::core_registration_callback,
                                                 true /*global*/);
}
}
