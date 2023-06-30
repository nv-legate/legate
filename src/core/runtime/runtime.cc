/* Copyright 2021-2023 NVIDIA Corporation
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

#include "core/runtime/runtime.h"

#include "core/operation/detail/copy.h"
#include "core/operation/detail/operation.h"
#include "core/operation/detail/task.h"
#include "core/runtime/context.h"
#include "core/runtime/detail/runtime.h"

namespace legate {

extern Logger log_legate;

// This is the unique string name for our library which can be used
// from both C++ and Python to generate IDs
extern const char* const core_library_name;

/*static*/ bool Core::show_progress_requested = false;

/*static*/ bool Core::use_empty_task = false;

/*static*/ bool Core::synchronize_stream_view = false;

/*static*/ bool Core::log_mapping_decisions = false;

/*static*/ bool Core::has_socket_mem = false;

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

/*static*/ void Core::perform_callback(Legion::RegistrationCallbackFnptr callback)
{
  Legion::Runtime::perform_registration_callback(callback, true /*global*/);
}

LibraryContext* Runtime::find_library(const std::string& library_name,
                                      bool can_fail /*=false*/) const
{
  return impl_->find_library(library_name, can_fail);
}

LibraryContext* Runtime::create_library(const std::string& library_name,
                                        const ResourceConfig& config,
                                        std::unique_ptr<mapping::Mapper> mapper)
{
  return impl_->create_library(library_name, config, std::move(mapper));
}

void Runtime::record_reduction_operator(int32_t type_uid, int32_t op_kind, int32_t legion_op_id)
{
  impl_->record_reduction_operator(type_uid, op_kind, legion_op_id);
}

// This function should be moved to the library context
AutoTask Runtime::create_task(LibraryContext* library, int64_t task_id)
{
  return AutoTask(impl_->create_task(library, task_id));
}

ManualTask Runtime::create_task(LibraryContext* library, int64_t task_id, const Shape& launch_shape)
{
  return ManualTask(impl_->create_task(library, task_id, launch_shape));
}

void Runtime::issue_copy(LogicalStore target, LogicalStore source)
{
  impl_->issue_copy(target.impl(), source.impl());
}

void Runtime::issue_gather(LogicalStore target, LogicalStore source, LogicalStore source_indirect)
{
  impl_->issue_gather(target.impl(), source.impl(), source_indirect.impl());
}

void Runtime::issue_scatter(LogicalStore target, LogicalStore target_indirect, LogicalStore source)
{
  impl_->issue_scatter(target.impl(), target_indirect.impl(), source.impl());
}

void Runtime::issue_scatter_gather(LogicalStore target,
                                   LogicalStore target_indirect,
                                   LogicalStore source,
                                   LogicalStore source_indirect)
{
  impl_->issue_scatter_gather(
    target.impl(), target_indirect.impl(), source.impl(), source_indirect.impl());
}

void Runtime::issue_fill(LogicalStore lhs, LogicalStore value)
{
  impl_->issue_fill(lhs.impl(), value.impl());
}

void Runtime::issue_fill(LogicalStore lhs, const Scalar& value)
{
  issue_fill(std::move(lhs), create_store(value));
}

void Runtime::submit(AutoTask&& task) { impl_->submit(std::move(task.impl_)); }

void Runtime::submit(ManualTask&& task) { impl_->submit(std::move(task.impl_)); }

LogicalStore Runtime::create_store(std::unique_ptr<Type> type, int32_t dim)
{
  return LogicalStore(impl_->create_store(std::move(type), dim));
}

LogicalStore Runtime::create_store(const Type& type, int32_t dim)
{
  return create_store(type.clone(), dim);
}

LogicalStore Runtime::create_store(const Shape& extents,
                                   std::unique_ptr<Type> type,
                                   bool optimize_scalar /*=false*/)
{
  return LogicalStore(impl_->create_store(extents, std::move(type), optimize_scalar));
}

LogicalStore Runtime::create_store(const Shape& extents,
                                   const Type& type,
                                   bool optimize_scalar /*=false*/)
{
  return create_store(extents, type.clone(), optimize_scalar);
}

LogicalStore Runtime::create_store(const Scalar& scalar)
{
  return LogicalStore(impl_->create_store(scalar));
}

uint32_t Runtime::max_pending_exceptions() const { return impl_->max_pending_exceptions(); }

void Runtime::set_max_pending_exceptions(uint32_t max_pending_exceptions)
{
  impl_->set_max_pending_exceptions(max_pending_exceptions);
}

void Runtime::raise_pending_task_exception() { impl_->raise_pending_task_exception(); }

std::optional<TaskException> Runtime::check_pending_task_exception()
{
  return impl_->check_pending_task_exception();
}

void Runtime::issue_execution_fence(bool block /*=false*/) { impl_->issue_execution_fence(block); }

const mapping::MachineDesc& Runtime::get_machine() const { return impl_->get_machine(); }

/*static*/ Runtime* Runtime::get_runtime()
{
  static Runtime* the_runtime{nullptr};
  if (nullptr == the_runtime) {
    auto* impl = detail::Runtime::get_runtime();
    if (!impl->initialized())
      throw std::runtime_error(
        "Legate runtime has not been initialized. Please invoke legate::start to use the runtime");

    the_runtime = new Runtime(impl);
  }
  return the_runtime;
}

Runtime::~Runtime() {}

Runtime::Runtime(detail::Runtime* impl) : impl_(impl) {}

int32_t start(int32_t argc, char** argv) { return detail::Runtime::start(argc, argv); }

int32_t finish() { return detail::Runtime::get_runtime()->finish(); }

const mapping::MachineDesc& get_machine() { return Runtime::get_runtime()->get_machine(); }

}  // namespace legate

extern "C" {

void legate_core_perform_registration()
{
  // Tell the runtime about our registration callback so we can register ourselves
  // Make sure it is global so this shared object always gets loaded on all nodes
  Legion::Runtime::perform_registration_callback(legate::detail::registration_callback_for_python,
                                                 true /*global*/);
}
}
