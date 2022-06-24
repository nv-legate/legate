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

#include "core/runtime/launcher.h"
#include "core/data/logical_store.h"
#include "core/data/scalar.h"
#include "core/runtime/context.h"
#include "core/runtime/runtime.h"
#include "core/utilities/buffer_builder.h"
#include "core/utilities/dispatch.h"

namespace legate {

class RequirementAnalyzer {
 public:
  ~RequirementAnalyzer();

 public:
  void insert(RegionReq* req, Legion::FieldID field_id);
  uint32_t get_requirement_index(RegionReq* req, Legion::FieldID field_id) const;

 public:
  void analyze_requirements();
  template <typename Task>
  void populate_launcher(Task* task) const;

 private:
  std::map<RegionReq*, uint32_t> req_indices_;
  std::vector<std::pair<RegionReq*, std::vector<Legion::FieldID>>> requirements_;
};

struct ArgWrapper {
  virtual ~ArgWrapper() {}
  virtual void pack(BufferBuilder& buffer) const = 0;
};

template <typename T>
struct ScalarArg : public ArgWrapper {
 public:
  ScalarArg(const T& value) : value_(value) {}

 public:
  virtual ~ScalarArg() {}

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 private:
  T value_;
};

struct UntypedScalarArg : public ArgWrapper {
 public:
  UntypedScalarArg(const Scalar& scalar) : scalar_(scalar) {}

 public:
  virtual ~UntypedScalarArg() {}

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 private:
  Scalar scalar_;
};

struct RegionFieldArg : public ArgWrapper {
 public:
  RegionFieldArg(RequirementAnalyzer* analyzer,
                 LogicalStore store,
                 int32_t dim,
                 RegionReq* req,
                 Legion::FieldID field_id,
                 Legion::ReductionOpID redop);

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 public:
  virtual ~RegionFieldArg() {}

 private:
  RequirementAnalyzer* analyzer_;
  LogicalStore store_;
  int32_t dim_;
  RegionReq* req_;
  Legion::FieldID field_id_;
  Legion::ReductionOpID redop_;
};

struct FutureStoreArg : public ArgWrapper {
 public:
  FutureStoreArg(LogicalStore store, bool read_only, bool has_storage, Legion::ReductionOpID redop);

 public:
  virtual ~FutureStoreArg() {}

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 private:
  LogicalStore store_;
  bool read_only_;
  bool has_storage_;
  Legion::ReductionOpID redop_;
};

RequirementAnalyzer::~RequirementAnalyzer()
{
  for (auto& pair : requirements_) delete pair.first;
}

void RequirementAnalyzer::insert(RegionReq* req, Legion::FieldID field_id)
{
  uint32_t req_idx = static_cast<uint32_t>(requirements_.size());
  requirements_.push_back(std::make_pair(req, std::vector<Legion::FieldID>({field_id})));
  req_indices_[req] = req_idx;
}

uint32_t RequirementAnalyzer::get_requirement_index(RegionReq* req, Legion::FieldID field_id) const
{
  auto finder = req_indices_.find(req);
  assert(finder != req_indices_.end());
  return finder->second;
}

void RequirementAnalyzer::analyze_requirements() {}

template <typename Task>
void RequirementAnalyzer::populate_launcher(Task* task) const
{
  for (auto& pair : requirements_) {
    auto& req    = pair.first;
    auto& fields = pair.second;
    req->proj->populate_launcher(task, *req, fields);
  }
}

template <typename T>
void ScalarArg<T>::pack(BufferBuilder& buffer) const
{
  buffer.pack(value_);
}

void UntypedScalarArg::pack(BufferBuilder& buffer) const
{
  buffer.pack<bool>(scalar_.is_tuple());
  buffer.pack<int32_t>(scalar_.code());
  buffer.pack_buffer(scalar_.ptr(), scalar_.size());
}

RegionFieldArg::RegionFieldArg(RequirementAnalyzer* analyzer,
                               LogicalStore store,
                               int32_t dim,
                               RegionReq* req,
                               Legion::FieldID field_id,
                               Legion::ReductionOpID redop)
  : analyzer_(analyzer),
    store_(std::move(store)),
    dim_(dim),
    req_(req),
    field_id_(field_id),
    redop_(redop)
{
}

void RegionFieldArg::pack(BufferBuilder& buffer) const
{
  store_.pack(buffer);

  buffer.pack<int32_t>(redop_);
  buffer.pack<int32_t>(dim_);
  buffer.pack<uint32_t>(analyzer_->get_requirement_index(req_, field_id_));
  buffer.pack<uint32_t>(field_id_);
}

FutureStoreArg::FutureStoreArg(LogicalStore store,
                               bool read_only,
                               bool has_storage,
                               Legion::ReductionOpID redop)
  : store_(std::move(store)), read_only_(read_only), has_storage_(has_storage), redop_(redop)
{
}

struct datalen_fn {
  template <LegateTypeCode CODE>
  size_t operator()()
  {
    return sizeof(legate_type_of<CODE>);
  }
};

void FutureStoreArg::pack(BufferBuilder& buffer) const
{
  store_.pack(buffer);

  buffer.pack<int32_t>(redop_);
  buffer.pack<bool>(read_only_);
  buffer.pack<bool>(has_storage_);
  buffer.pack<int32_t>(type_dispatch(store_.code(), datalen_fn{}));
  buffer.pack<size_t>(store_.extents());
}

Projection::Projection(Legion::ReductionOpID r) : redop(r) {}

void Projection::set_reduction_op(Legion::ReductionOpID r) { redop = r; }

Replicate::Replicate() : Projection() {}

Replicate::Replicate(Legion::ReductionOpID redop) : Projection(redop) {}

void Replicate::populate_launcher(Legion::TaskLauncher* task,
                                  const RegionReq& req,
                                  const std::vector<Legion::FieldID>& fields) const
{
  if (req.priv == REDUCE) {
#ifdef DEBUG_LEGATE
    assert(redop != -1);
#endif
    Legion::RegionRequirement legion_req(req.region, redop, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  } else {
    Legion::RegionRequirement legion_req(req.region, req.priv, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  }
}

void Replicate::populate_launcher(Legion::IndexTaskLauncher* task,
                                  const RegionReq& req,
                                  const std::vector<Legion::FieldID>& fields) const
{
  if (req.priv == REDUCE) {
#ifdef DEBUG_LEGATE
    assert(redop != -1);
#endif
    Legion::RegionRequirement legion_req(req.region, redop, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  } else {
    Legion::RegionRequirement legion_req(req.region, req.priv, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  }
}

MapPartition::MapPartition(Legion::LogicalPartition partition, Legion::ProjectionID proj_id)
  : Projection(), partition_(partition), proj_id_(proj_id)
{
}

MapPartition::MapPartition(Legion::LogicalPartition partition,
                           Legion::ProjectionID proj_id,
                           Legion::ReductionOpID redop)
  : Projection(redop), partition_(partition), proj_id_(proj_id)
{
}

void MapPartition::populate_launcher(Legion::TaskLauncher* task,
                                     const RegionReq& req,
                                     const std::vector<Legion::FieldID>& fields) const
{
  if (req.priv == REDUCE) {
#ifdef DEBUG_LEGATE
    assert(redop != -1);
#endif
    Legion::RegionRequirement legion_req(req.region, redop, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  } else {
    Legion::RegionRequirement legion_req(req.region, req.priv, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  }
}

void MapPartition::populate_launcher(Legion::IndexTaskLauncher* task,
                                     const RegionReq& req,
                                     const std::vector<Legion::FieldID>& fields) const
{
  if (req.priv == REDUCE) {
#ifdef DEBUG_LEGATE
    assert(redop != -1);
#endif
    Legion::RegionRequirement legion_req(
      partition_, proj_id_, redop, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  } else {
    Legion::RegionRequirement legion_req(
      partition_, proj_id_, req.priv, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  }
}

RegionReq::RegionReq(Legion::LogicalRegion _region,
                     Legion::PrivilegeMode _priv,
                     ProjectionP _proj,
                     int64_t _tag)
  : region(_region), priv(_priv), proj(std::move(_proj)), tag(_tag)
{
}

TaskLauncher::TaskLauncher(Runtime* runtime,
                           LibraryContext* library,
                           int64_t task_id,
                           int64_t mapper_id /*= 0*/,
                           int64_t tag /*= 0*/)
  : runtime_(runtime), library_(library), task_id_(task_id), mapper_id_(mapper_id), tag_(tag)
{
  req_analyzer_ = new RequirementAnalyzer();
  buffer_       = new BufferBuilder();
}

TaskLauncher::~TaskLauncher()
{
  for (auto& arg : inputs_) delete arg;
  for (auto& arg : outputs_) delete arg;
  for (auto& arg : reductions_) delete arg;
  for (auto& arg : scalars_) delete arg;

  delete req_analyzer_;
  delete buffer_;
}

int64_t TaskLauncher::legion_task_id() const { return library_->get_task_id(task_id_); }

int64_t TaskLauncher::legion_mapper_id() const { return library_->get_mapper_id(mapper_id_); }

void TaskLauncher::add_scalar(const Scalar& scalar)
{
  scalars_.push_back(new UntypedScalarArg(scalar));
}

void TaskLauncher::add_input(LogicalStore store, ProjectionP proj, uint64_t tag /*= 0*/)
{
  add_store(inputs_, std::move(store), std::move(proj), READ_ONLY, tag);
}

void TaskLauncher::add_output(LogicalStore store, ProjectionP proj, uint64_t tag /*= 0*/)
{
  add_store(outputs_, std::move(store), std::move(proj), WRITE_ONLY, tag);
}

void TaskLauncher::add_reduction(LogicalStore store,
                                 ProjectionP proj,
                                 uint64_t tag /*= 0*/,
                                 bool read_write /*= false*/)
{
  assert(!read_write);
  add_store(reductions_, std::move(store), std::move(proj), REDUCE, tag);
}

void TaskLauncher::execute(const Legion::Domain& launch_domain)
{
  auto legion_launcher = build_index_task(launch_domain);
  runtime_->dispatch(legion_launcher);
  delete legion_launcher;
}

void TaskLauncher::execute_single()
{
  auto legion_launcher = build_single_task();
  runtime_->dispatch(legion_launcher);
  delete legion_launcher;
}

void TaskLauncher::add_store(std::vector<ArgWrapper*>& args,
                             LogicalStore store,
                             ProjectionP proj,
                             Legion::PrivilegeMode privilege,
                             uint64_t tag)
{
  auto redop = proj->redop;

  if (store.scalar()) {
    auto has_storage = privilege != WRITE_ONLY;
    auto read_only   = privilege == READ_ONLY;
    if (has_storage) futures_.push_back(store.get_future());
    args.push_back(new FutureStoreArg(std::move(store), read_only, has_storage, redop));
  } else {
    auto storage  = store.get_storage();
    auto region   = storage->region();
    auto field_id = storage->field_id();

    auto req = new RegionReq(region, privilege, std::move(proj), tag);

    req_analyzer_->insert(req, field_id);
    args.push_back(
      new RegionFieldArg(req_analyzer_, std::move(store), region.get_dim(), req, field_id, redop));
  }
}

void TaskLauncher::pack_args(const std::vector<ArgWrapper*>& args)
{
  buffer_->pack<uint32_t>(args.size());
  for (auto& arg : args) arg->pack(*buffer_);
}

Legion::TaskLauncher* TaskLauncher::build_single_task()
{
  pack_args(inputs_);
  pack_args(outputs_);
  pack_args(reductions_);
  pack_args(scalars_);
  buffer_->pack<bool>(false);
  buffer_->pack<bool>(false);
  buffer_->pack<uint32_t>(0);

  auto single_task = new Legion::TaskLauncher(legion_task_id(),
                                              buffer_->to_legion_buffer(),
                                              Legion::Predicate::TRUE_PRED,
                                              legion_mapper_id(),
                                              tag_);
  for (auto& future_ : futures_) single_task->add_future(future_);

  req_analyzer_->populate_launcher(single_task);

  return single_task;
}

Legion::IndexTaskLauncher* TaskLauncher::build_index_task(const Legion::Domain& launch_domain)
{
  pack_args(inputs_);
  pack_args(outputs_);
  pack_args(reductions_);
  pack_args(scalars_);
  buffer_->pack<bool>(false);
  buffer_->pack<bool>(false);
  buffer_->pack<uint32_t>(0);

  auto index_task = new Legion::IndexTaskLauncher(legion_task_id(),
                                                  launch_domain,
                                                  buffer_->to_legion_buffer(),
                                                  Legion::ArgumentMap(),
                                                  Legion::Predicate::TRUE_PRED,
                                                  false /*must*/,
                                                  legion_mapper_id(),
                                                  tag_);
  for (auto& future_ : futures_) index_task->add_future(future_);

  req_analyzer_->populate_launcher(index_task);

  return index_task;
}

}  // namespace legate
