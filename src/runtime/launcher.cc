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

#include "runtime/launcher.h"
#include "data/logical_store.h"
#include "data/scalar.h"
#include "runtime/context.h"
#include "runtime/runtime.h"

namespace legate {

class BufferBuilder {
 public:
  BufferBuilder();

 public:
  template <typename T>
  void pack(const T& value);
  void pack_buffer(const void* buffer, size_t size);

 public:
  Legion::UntypedBuffer to_legion_buffer() const;

 private:
  std::vector<int8_t> buffer_;
};

class RequirementAnalyzer {
 private:
  using SingleTask = Legion::TaskLauncher*;

 public:
  ~RequirementAnalyzer();

 public:
  void insert(RegionReq* req, Legion::FieldID field_id);
  uint32_t get_requirement_index(RegionReq* req, Legion::FieldID field_id) const;

 public:
  void analyze_requirements();
  void populate_launcher(SingleTask task) const;

 private:
  std::map<RegionReq*, uint32_t> req_indices_;
  std::vector<std::pair<RegionReq*, std::vector<Legion::FieldID>>> requirements_;
};

struct ArgWrapper {
  virtual void pack(BufferBuilder& buffer) const = 0;
};

template <typename T>
struct ScalarArg : public ArgWrapper {
 public:
  ScalarArg(const T& value) : value_(value) {}

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 private:
  T value_;
};

struct UntypedScalarArg : public ArgWrapper {
 public:
  UntypedScalarArg(const Scalar& scalar) : scalar_(scalar) {}

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 private:
  Scalar scalar_;
};

struct RegionFieldArg : public ArgWrapper {
 private:
  using LogicalStoreP = std::shared_ptr<LogicalStore>;

 public:
  RegionFieldArg(RequirementAnalyzer* analyzer,
                 LogicalStoreP store,
                 int32_t dim,
                 RegionReq* req,
                 Legion::FieldID field_id,
                 Legion::ReductionOpID redop);

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 private:
  RequirementAnalyzer* analyzer_;
  LogicalStoreP store_;
  int32_t dim_;
  RegionReq* req_;
  Legion::FieldID field_id_;
  Legion::ReductionOpID redop_;
};

BufferBuilder::BufferBuilder()
{
  // Reserve 4KB to minimize resizing while packing the arguments.
  buffer_.reserve(4096);
}

template <typename T>
void BufferBuilder::pack(const T& value)
{
  pack_buffer(reinterpret_cast<const int8_t*>(&value), sizeof(T));
}

void BufferBuilder::pack_buffer(const void* src, size_t size)
{
  auto tgt = buffer_.data() + buffer_.size();
  buffer_.resize(buffer_.size() + size);
  memcpy(tgt, src, size);
}

Legion::UntypedBuffer BufferBuilder::to_legion_buffer() const
{
  return Legion::UntypedBuffer(buffer_.data(), buffer_.size());
}

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

void RequirementAnalyzer::populate_launcher(SingleTask task) const
{
  for (auto& pair : requirements_) {
    auto& req    = pair.first;
    auto& fields = pair.second;
    req->proj->add(task, *req, fields);
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
                               LogicalStoreP store,
                               int32_t dim,
                               RegionReq* req,
                               Legion::FieldID field_id,
                               Legion::ReductionOpID redop)
  : analyzer_(analyzer), store_(store), dim_(dim), req_(req), field_id_(field_id), redop_(redop)
{
}

void RegionFieldArg::pack(BufferBuilder& buffer) const
{
  buffer.pack<bool>(false);
  buffer.pack<int32_t>(store_->dim());
  buffer.pack<int32_t>(store_->code());
  buffer.pack<int32_t>(-1);

  buffer.pack<int32_t>(redop_);
  buffer.pack<int32_t>(dim_);
  buffer.pack<uint32_t>(analyzer_->get_requirement_index(req_, field_id_));
  buffer.pack<uint32_t>(field_id_);
}

Projection::Projection(Legion::ReductionOpID r) : redop(std::make_unique<Legion::ReductionOpID>(r))
{
}

Broadcast::Broadcast() : Projection() {}

Broadcast::Broadcast(Legion::ReductionOpID redop) : Projection(redop) {}
void Broadcast::add(SingleTask task,
                    const RegionReq& req,
                    const std::vector<Legion::FieldID>& fields) const
{
  if (req.priv == REDUCE) {
    Legion::RegionRequirement legion_req(req.region, *redop, EXCLUSIVE, req.region, req.tag);
    legion_req.add_fields(fields);
    task->add_region_requirement(legion_req);
  } else {
    Legion::RegionRequirement legion_req(req.region, req.priv, EXCLUSIVE, req.region, req.tag);
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

void TaskLauncher::add_input(LogicalStoreP store, ProjectionP proj, uint64_t tag /*= 0*/)
{
  add_store(inputs_, std::move(store), std::move(proj), READ_ONLY, tag);
}

void TaskLauncher::add_output(LogicalStoreP store, ProjectionP proj, uint64_t tag /*= 0*/)
{
  add_store(outputs_, std::move(store), std::move(proj), WRITE_ONLY, tag);
}

void TaskLauncher::add_reduction(LogicalStoreP store,
                                 ProjectionP proj,
                                 uint64_t tag /*= 0*/,
                                 bool read_write /*= false*/)
{
  assert(!read_write);
  add_store(reductions_, std::move(store), std::move(proj), REDUCE, tag);
}

void TaskLauncher::execute_single()
{
  auto legion_launcher = build_single_task();
  runtime_->dispatch(legion_launcher);
  delete legion_launcher;
}

void TaskLauncher::add_store(std::vector<ArgWrapper*>& args,
                             LogicalStoreP store,
                             ProjectionP proj,
                             Legion::PrivilegeMode privilege,
                             uint64_t tag)
{
  auto storage  = store->get_storage();
  auto region   = storage->region();
  auto field_id = storage->field_id();

  auto redop = nullptr != proj->redop ? *proj->redop : -1;
  auto req   = new RegionReq(region, privilege, std::move(proj), tag);

  req_analyzer_->insert(req, field_id);
  args.push_back(
    new RegionFieldArg(req_analyzer_, std::move(store), region.get_dim(), req, field_id, redop));
}

void TaskLauncher::pack_args(const std::vector<ArgWrapper*>& args)
{
  buffer_->pack<uint32_t>(args.size());
  for (auto& arg : args) arg->pack(*buffer_);
}

TaskLauncher::SingleTask TaskLauncher::build_single_task()
{
  pack_args(inputs_);
  pack_args(outputs_);
  pack_args(reductions_);
  pack_args(scalars_);

  auto single_task = new Legion::TaskLauncher(legion_task_id(),
                                              buffer_->to_legion_buffer(),
                                              Legion::Predicate::TRUE_PRED,
                                              legion_mapper_id(),
                                              tag_);

  req_analyzer_->populate_launcher(single_task);

  return single_task;
}

}  // namespace legate
