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

#include "mappers/null_mapper.h"

#include "legate.h"

#include "core/mapping/core_mapper.h"

namespace legate {

using namespace Legion;
using namespace Legion::Mapping;

uint32_t extract_env(const char* env_name, const uint32_t default_value, const uint32_t test_value)
{
  const char* env_value = getenv(env_name);
  if (env_value == NULL) {
    const char* legate_test = getenv("LEGATE_TEST");
    if (legate_test != NULL)
      return test_value;
    else
      return default_value;
  } else
    return atoi(env_value);
}

// This is a custom mapper implementation that only has to map
// start-up tasks associated with the Legate core, no one else
// should be overriding this mapper so we burry it in here
class CoreMapper : public Legion::Mapping::NullMapper {
 public:
  CoreMapper(MapperRuntime* runtime, Machine machine, const LibraryContext& context);
  virtual ~CoreMapper(void);

 public:
  // Start-up methods
  static AddressSpaceID get_local_node(void);
  static size_t get_total_nodes(Machine m);
  static const char* create_name(AddressSpace node);

 public:
  virtual const char* get_mapper_name(void) const;
  virtual MapperSyncModel get_mapper_sync_model(void) const;
  virtual bool request_valid_instances(void) const { return false; }

 public:  // Task mapping calls
  virtual void select_task_options(const MapperContext ctx, const Task& task, TaskOptions& output);
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  virtual void map_task(const MapperContext ctx,
                        const Task& task,
                        const MapTaskInput& input,
                        MapTaskOutput& output);
  virtual void select_sharding_functor(const MapperContext ctx,
                                       const Task& task,
                                       const SelectShardingFunctorInput& input,
                                       SelectShardingFunctorOutput& output);
  virtual void select_steal_targets(const MapperContext ctx,
                                    const SelectStealingInput& input,
                                    SelectStealingOutput& output);
  virtual void select_tasks_to_map(const MapperContext ctx,
                                   const SelectMappingInput& input,
                                   SelectMappingOutput& output);

 public:
  virtual void configure_context(const MapperContext ctx,
                                 const Task& task,
                                 ContextConfigOutput& output);
  void map_future_map_reduction(const MapperContext ctx,
                                const FutureMapReductionInput& input,
                                FutureMapReductionOutput& output);
  virtual void select_tunable_value(const MapperContext ctx,
                                    const Task& task,
                                    const SelectTunableInput& input,
                                    SelectTunableOutput& output);
  void pack_tunable(const int value, Mapper::SelectTunableOutput& output);

 public:
  const AddressSpace local_node;
  const size_t total_nodes;
  const char* const mapper_name;
  LibraryContext context;

 protected:
  const unsigned min_gpu_chunk;
  const unsigned min_cpu_chunk;
  const unsigned min_omp_chunk;
  const unsigned window_size;

 protected:
  std::vector<Processor> local_cpus;
  std::vector<Processor> local_omps;
  std::vector<Processor> local_gpus;

 protected:
  Memory local_system_memory, local_zerocopy_memory;
  std::map<Processor, Memory> local_frame_buffers;
};

CoreMapper::CoreMapper(MapperRuntime* rt, Machine m, const LibraryContext& c)
  : NullMapper(rt, m),
    local_node(get_local_node()),
    total_nodes(get_total_nodes(m)),
    mapper_name(create_name(local_node)),
    context(c),
    min_gpu_chunk(extract_env("LEGATE_MIN_GPU_CHUNK", 1 << 20, 2)),
    min_cpu_chunk(extract_env("LEGATE_MIN_CPU_CHUNK", 1 << 14, 2)),
    min_omp_chunk(extract_env("LEGATE_MIN_OMP_CHUNK", 1 << 17, 2)),
    window_size(extract_env("LEGATE_WINDOW_SIZE", 1, 1))
{
  // Query to find all our local processors
  Machine::ProcessorQuery local_procs(machine);
  local_procs.local_address_space();
  for (Machine::ProcessorQuery::iterator it = local_procs.begin(); it != local_procs.end(); it++) {
    switch (it->kind()) {
      case Processor::LOC_PROC: {
        local_cpus.push_back(*it);
        break;
      }
      case Processor::OMP_PROC: {
        local_omps.push_back(*it);
        break;
      }
      case Processor::TOC_PROC: {
        local_gpus.push_back(*it);
        break;
      }
      default: break;
    }
  }
  // Now do queries to find all our local memories
  Machine::MemoryQuery local_sysmem(machine);
  local_sysmem.local_address_space();
  local_sysmem.only_kind(Memory::SYSTEM_MEM);
  assert(local_sysmem.count() > 0);
  local_system_memory = local_sysmem.first();
  if (!local_gpus.empty()) {
    Machine::MemoryQuery local_zcmem(machine);
    local_zcmem.local_address_space();
    local_zcmem.only_kind(Memory::Z_COPY_MEM);
    assert(local_zcmem.count() > 0);
    local_zerocopy_memory = local_zcmem.first();
  }
  for (std::vector<Processor>::const_iterator it = local_gpus.begin(); it != local_gpus.end();
       it++) {
    Machine::MemoryQuery local_framebuffer(machine);
    local_framebuffer.local_address_space();
    local_framebuffer.only_kind(Memory::GPU_FB_MEM);
    local_framebuffer.best_affinity_to(*it);
    assert(local_framebuffer.count() > 0);
    local_frame_buffers[*it] = local_framebuffer.first();
  }
}

CoreMapper::~CoreMapper(void) { free(const_cast<char*>(mapper_name)); }

/*static*/ AddressSpace CoreMapper::get_local_node(void)
{
  Processor p = Processor::get_executing_processor();
  return p.address_space();
}

/*static*/ size_t CoreMapper::get_total_nodes(Machine m)
{
  Machine::ProcessorQuery query(m);
  query.only_kind(Processor::LOC_PROC);
  std::set<AddressSpace> spaces;
  for (Machine::ProcessorQuery::iterator it = query.begin(); it != query.end(); it++)
    spaces.insert(it->address_space());
  return spaces.size();
}

/*static*/ const char* CoreMapper::create_name(AddressSpace node)
{
  char buffer[128];
  snprintf(buffer, 127, "Legate Mapper on Node %d", node);
  return strdup(buffer);
}

const char* CoreMapper::get_mapper_name(void) const { return mapper_name; }

Mapper::MapperSyncModel CoreMapper::get_mapper_sync_model(void) const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

void CoreMapper::select_task_options(const MapperContext ctx, const Task& task, TaskOptions& output)
{
  std::cout<<"task_id "<<task.task_id<<std::endl;
  //assert(context.valid_task_id(task.task_id));
  if (task.tag == LEGATE_CPU_VARIANT) {
    assert(!local_cpus.empty());
    output.initial_proc = local_cpus.front();
  } else {
    assert(task.tag == LEGATE_GPU_VARIANT);
    assert(!local_gpus.empty());
    output.initial_proc = local_gpus.front();
  }
}

void CoreMapper::slice_task(const MapperContext ctx,
                            const Task& task,
                            const SliceTaskInput& input,
                            SliceTaskOutput& output)
{
  assert(context.valid_task_id(task.task_id));
  output.slices.reserve(input.domain.get_volume());
  // Check to see if we're control replicated or not. If we are then
  // we'll already have been sharded.
  Machine::ProcessorQuery all_procs(machine);
  all_procs.only_kind(task.target_proc.kind());
  if (all_procs.count() == input.domain.get_volume()) {
    Machine::ProcessorQuery::iterator pit = all_procs.begin();
    for (Domain::DomainPointIterator itr(input.domain); itr; itr++, pit++)
      output.slices.push_back(
        TaskSlice(Domain(itr.p, itr.p), *pit, false /*recurse*/, false /*stealable*/));
  } else {
    // Control-replicated because we've already been sharded
    Domain sharding_domain = task.index_domain;
    if (task.sharding_space.exists())
      sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);
    assert(sharding_domain.get_dim() == 1);
    assert(input.domain.get_dim() == 1);
    const Rect<1> space = sharding_domain;
    const Rect<1> local = input.domain;
    const size_t size   = (space.hi[0] - space.lo[0]) + 1;
    // Assume that if we're control replicated there is one shard per space
    const coord_t chunk = (size + total_nodes - 1) / total_nodes;
    const coord_t start = local_node * chunk + space.lo[0];
    switch (task.target_proc.kind()) {
      case Processor::LOC_PROC: {
        for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
          const Point<1> point = itr.p;
          assert(point[0] >= start);
          assert(point[0] < (start + chunk));
          const unsigned local_index = point[0] - start;
          assert(local_index < local_cpus.size());
          output.slices.push_back(TaskSlice(
            Domain(itr.p, itr.p), local_cpus[local_index], false /*recurse*/, false /*stealable*/));
        }
        break;
      }
      case Processor::TOC_PROC: {
        for (Domain::DomainPointIterator itr(input.domain); itr; itr++) {
          const Point<1> point = itr.p;
          assert(point[0] >= start);
          assert(point[0] < (start + chunk));
          const unsigned local_index = point[0] - start;
          assert(local_index < local_gpus.size());
          output.slices.push_back(TaskSlice(
            Domain(itr.p, itr.p), local_gpus[local_index], false /*recurse*/, false /*stealable*/));
        }
        break;
      }
      default: LEGATE_ABORT
    }
  }
}

void CoreMapper::map_task(const MapperContext ctx,
                          const Task& task,
                          const MapTaskInput& input,
                          MapTaskOutput& output)
{
  assert(context.valid_task_id(task.task_id));
  // Just put our target proc in the target processors for now
  output.target_procs.push_back(task.target_proc);
  output.chosen_variant = task.tag;
}

void CoreMapper::select_sharding_functor(const MapperContext ctx,
                                         const Task& task,
                                         const SelectShardingFunctorInput& input,
                                         SelectShardingFunctorOutput& output)
{
  assert(context.valid_task_id(task.task_id));
  assert(task.regions.empty());
  const int launch_dim = task.index_domain.get_dim();
  assert(launch_dim == 1);
  output.chosen_functor = context.get_sharding_id(0);
}

void CoreMapper::select_steal_targets(const MapperContext ctx,
                                      const SelectStealingInput& input,
                                      SelectStealingOutput& output)
{
  // Do nothing
}

void CoreMapper::select_tasks_to_map(const MapperContext ctx,
                                     const SelectMappingInput& input,
                                     SelectMappingOutput& output)
{
  output.map_tasks.insert(input.ready_tasks.begin(), input.ready_tasks.end());
}

void CoreMapper::configure_context(const MapperContext ctx,
                                   const Task& task,
                                   ContextConfigOutput& output)
{
  // Use the defaults currently
}

void CoreMapper::pack_tunable(const int value, Mapper::SelectTunableOutput& output)
{
  int* result  = (int*)malloc(sizeof(value));
  *result      = value;
  output.value = result;
  output.size  = sizeof(value);
}

void CoreMapper::map_future_map_reduction(const MapperContext ctx,
                                          const FutureMapReductionInput& input,
                                          FutureMapReductionOutput& output)
{
}

void CoreMapper::select_tunable_value(const MapperContext ctx,
                                      const Task& task,
                                      const SelectTunableInput& input,
                                      SelectTunableOutput& output)
{
  switch (input.tunable_id) {
    case LEGATE_CORE_TUNABLE_TOTAL_CPUS: {
      pack_tunable(local_cpus.size() * total_nodes, output);  // assume symmetry
      return;
    }
    case LEGATE_CORE_TUNABLE_TOTAL_GPUS: {
      pack_tunable(local_gpus.size() * total_nodes, output);  // assume symmetry
      return;
    }
    case LEGATE_CORE_TUNABLE_NUM_PIECES: {
      if (!local_gpus.empty())  // If we have GPUs, use those
        pack_tunable(local_gpus.size() * total_nodes, output);
      else if (!local_omps.empty())  // Otherwise use OpenMP procs
        pack_tunable(local_omps.size() * total_nodes, output);
      else  // Otherwise use the CPUs
        pack_tunable(local_cpus.size() * total_nodes, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME: {
      // TODO: make these profile guided
      if (!local_gpus.empty())
        // Make sure we can get at least 1M elements on each GPU
        pack_tunable(min_gpu_chunk, output);
      else if (!local_omps.empty())
        // Make sure we get at least 128K elements on each OpenMP
        pack_tunable(min_omp_chunk, output);
      else
        // Make sure we can get at least 8KB elements on each CPU
        pack_tunable(min_cpu_chunk, output);
      return;
    }
    case LEGATE_CORE_TUNABLE_WINDOW_SIZE: {
      pack_tunable(window_size, output);
      return;
    }
  }
  // Illegal tunable variable
  LEGATE_ABORT
}

void register_legate_core_mapper(Machine machine, Runtime* runtime, const LibraryContext& context)
{
  // Replace all the default mappers with our custom mapper for the Legate
  // top-level task and init task
  runtime->add_mapper(context.get_mapper_id(0),
                      new CoreMapper(runtime->get_mapper_runtime(), machine, context));
}

}  // namespace legate
