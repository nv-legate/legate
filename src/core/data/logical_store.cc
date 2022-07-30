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

#include <numeric>

#include "core/data/logical_store.h"
#include "core/data/store.h"
#include "core/partitioning/partition.h"
#include "core/runtime/req_analyzer.h"
#include "core/runtime/runtime.h"
#include "core/utilities/buffer_builder.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/type_traits.h"
#include "legate_defines.h"

using namespace Legion;

namespace legate {

extern Logger log_legate;

LogicalRegionField::LogicalRegionField(Runtime* runtime, const LogicalRegion& lr, FieldID fid)
  : runtime_(runtime), lr_(lr), fid_(fid)
{
}

int32_t LogicalRegionField::dim() const { return lr_.get_dim(); }

Domain LogicalRegionField::domain() const
{
  return runtime_->get_index_space_domain(lr_.get_index_space());
}

namespace detail {

class LogicalStore {
 public:
  LogicalStore();
  LogicalStore(Runtime* runtime,
               LegateTypeCode code,
               tuple<size_t> extents,
               std::shared_ptr<LogicalStore> parent,
               std::shared_ptr<TransformStack> transform);
  LogicalStore(Runtime* runtime, LegateTypeCode code, const void* data);

 public:
  ~LogicalStore();

 private:
  LogicalStore(std::shared_ptr<detail::LogicalStore> impl);

 public:
  LogicalStore(const LogicalStore& other)            = default;
  LogicalStore& operator=(const LogicalStore& other) = default;

 public:
  LogicalStore(LogicalStore&& other)            = default;
  LogicalStore& operator=(LogicalStore&& other) = default;

 public:
  bool scalar() const;
  int32_t dim() const;
  LegateTypeCode code() const;
  Legion::Domain domain() const;
  const std::vector<size_t>& extents() const;
  size_t volume() const;

 public:
  bool has_storage() const;
  std::shared_ptr<LogicalRegionField> get_storage();
  Legion::Future get_future();

 private:
  void create_storage();

 public:
  std::shared_ptr<LogicalStore> promote(int32_t extra_dim,
                                        size_t dim_size,
                                        std::shared_ptr<LogicalStore> parent) const;

 public:
  std::shared_ptr<Store> get_physical_store(LibraryContext* context);

 public:
  std::unique_ptr<Projection> find_or_create_partition(const Partition* partition);
  std::unique_ptr<Partition> find_or_create_key_partition();

 private:
  std::unique_ptr<Partition> invert_partition(const Partition* partition) const;
  proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const;
  Legion::ProjectionID compute_projection() const;

 public:
  void pack(BufferBuilder& buffer) const;

 private:
  void pack_transform(BufferBuilder& buffer) const;

 private:
  bool scalar_{false};
  Runtime* runtime_{nullptr};
  LegateTypeCode code_{MAX_TYPE_NUMBER};
  tuple<size_t> extents_;
  std::shared_ptr<LogicalRegionField> region_field_{nullptr};
  Legion::Future future_{};
  std::shared_ptr<LogicalStore> parent_{nullptr};
  std::shared_ptr<TransformStack> transform_{nullptr};
  std::shared_ptr<Store> mapped_{nullptr};
};

LogicalStore::LogicalStore(Runtime* runtime,
                           LegateTypeCode code,
                           tuple<size_t> extents,
                           std::shared_ptr<LogicalStore> parent,
                           std::shared_ptr<TransformStack> transform)
  : runtime_(runtime),
    code_(code),
    extents_(std::move(extents)),
    parent_(std::move(parent)),
    transform_(std::move(transform))
{
}

struct datalen_fn {
  template <LegateTypeCode CODE>
  size_t operator()()
  {
    return sizeof(legate_type_of<CODE>);
  }
};

LogicalStore::LogicalStore(Runtime* runtime, LegateTypeCode code, const void* data)
  : scalar_(true), runtime_(runtime), code_(code), extents_({1})
{
  auto datalen = type_dispatch(code, datalen_fn{});
  future_      = runtime_->create_future(data, datalen);
}

LogicalStore::~LogicalStore()
{
  if (mapped_ != nullptr) mapped_->unmap();
}

bool LogicalStore::scalar() const { return scalar_; }

int32_t LogicalStore::dim() const { return static_cast<int32_t>(extents_.size()); }

LegateTypeCode LogicalStore::code() const { return code_; }

Domain LogicalStore::domain() const
{
  assert(nullptr != region_field_);
  return region_field_->domain();
}

const std::vector<size_t>& LogicalStore::extents() const { return extents_.data(); }

size_t LogicalStore::volume() const { return extents_.volume(); }

bool LogicalStore::has_storage() const { return nullptr != region_field_; }

std::shared_ptr<LogicalRegionField> LogicalStore::get_storage()
{
  assert(!scalar_);
  if (nullptr == parent_) {
    if (!has_storage()) create_storage();
    return region_field_;
  } else
    return parent_->get_storage();
}

Legion::Future LogicalStore::get_future()
{
  assert(scalar_);
  if (nullptr == parent_) {
    return future_;
  } else
    return parent_->get_future();
}

void LogicalStore::create_storage()
{
  region_field_ = runtime_->create_region_field(extents_, code_);
}

std::shared_ptr<LogicalStore> LogicalStore::promote(int32_t extra_dim,
                                                    size_t dim_size,
                                                    std::shared_ptr<LogicalStore> parent) const
{
  if (extra_dim < 0 || static_cast<size_t>(extra_dim) > extents_.size()) {
    log_legate.error(
      "Invalid promotion on dimension %d for a %zd-D store", extra_dim, extents_.size());
    LEGATE_ABORT;
  }

  auto new_extents = extents_.insert(extra_dim, dim_size);
  // TODO: Move this push operation to TransformStack.
  //       Two prerequisites:
  //         1) make members of TransformStack read only (i.e., by adding const)
  //         2) make TransformStack inherit std::enable_shared_from_this.
  //       Then we can add a const push method to TransformStack that returns
  //       a fresh shared_ptr of TransformStack with a transform put on top.
  auto transform =
    std::make_shared<TransformStack>(std::make_unique<Promote>(extra_dim, dim_size), transform_);
  return std::make_shared<LogicalStore>(
    runtime_, code_, std::move(new_extents), std::move(parent), std::move(transform));
}

std::shared_ptr<Store> LogicalStore::get_physical_store(LibraryContext* context)
{
  // TODO: Need to support inline mapping for scalars
  assert(!scalar_);
  if (nullptr != mapped_) return mapped_;
  auto rf = runtime_->map_region_field(context, region_field_);
  mapped_ = std::make_shared<Store>(dim(), code_, -1, std::move(rf), transform_);
  return mapped_;
}

std::unique_ptr<Partition> LogicalStore::invert_partition(const Partition* partition) const
{
  if (nullptr == parent_) {
    switch (partition->kind()) {
      case Partition::Kind::NO_PARTITION: {
        return create_no_partition(runtime_);
      }
      case Partition::Kind::TILING: {
        auto tiling       = static_cast<const Tiling*>(partition);
        Shape tile_shape  = tiling->tile_shape();
        Shape color_shape = tiling->color_shape();
        Shape offsets     = tiling->offsets();
        return create_tiling(
          runtime_, std::move(tile_shape), std::move(color_shape), std::move(offsets));
      }
    }
  } else {
    auto inverted = transform_->invert_partition(partition);
    return parent_->invert_partition(inverted.get());
  }
  assert(false);
  return nullptr;
}

proj::SymbolicPoint LogicalStore::invert(const proj::SymbolicPoint& point) const
{
  if (nullptr == parent_) return point;
  return parent_->invert(transform_->invert(point));
}

Legion::ProjectionID LogicalStore::compute_projection() const
{
  auto ndim = dim();
  std::vector<proj::SymbolicExpr> exprs;
  for (int32_t dim = 0; dim < ndim; ++dim) exprs.push_back(proj::SymbolicExpr(dim));
  auto point = invert(proj::SymbolicPoint(std::move(exprs)));

  bool identity_mapping = true;
  for (int32_t dim = 0; dim < ndim; ++dim)
    identity_mapping = identity_mapping && point[dim].is_identity(dim);

  if (identity_mapping)
    return 0;
  else
    return runtime_->get_projection(ndim, point);
}

std::unique_ptr<Projection> LogicalStore::find_or_create_partition(const Partition* partition)
{
  if (scalar_) return std::make_unique<Projection>();

  // We're about to create a legion partition for this store, so the store should have its region
  // created.
  auto proj     = compute_projection();
  auto inverted = invert_partition(partition);

  auto lr = get_storage()->region();
  auto lp =
    inverted->construct(lr, inverted->is_disjoint_for(nullptr), inverted->is_complete_for(nullptr));
  return std::make_unique<Projection>(lp, proj);
}

std::unique_ptr<Partition> LogicalStore::find_or_create_key_partition()
{
  if (scalar_) return create_no_partition(runtime_);

  auto part_mgr     = runtime_->partition_manager();
  auto launch_shape = part_mgr->compute_launch_shape(extents());
  if (launch_shape.empty())
    return create_no_partition(runtime_);
  else {
    auto tile_shape = part_mgr->compute_tile_shape(extents_, launch_shape);
    return create_tiling(runtime_, std::move(tile_shape), std::move(launch_shape));
  }
}

void LogicalStore::pack(BufferBuilder& buffer) const
{
  buffer.pack<bool>(scalar_);
  buffer.pack<bool>(false);
  buffer.pack<int32_t>(dim());
  buffer.pack<int32_t>(code_);
  if (transform_ != nullptr) transform_->pack(buffer);
  buffer.pack<int32_t>(-1);
}

void LogicalStore::pack_transform(BufferBuilder& buffer) const
{
  if (nullptr == parent_)
    buffer.pack<int32_t>(-1);
  else {
    transform_->pack(buffer);
    parent_->pack_transform(buffer);
  }
}

}  // namespace detail

LogicalStore::LogicalStore() {}

LogicalStore::LogicalStore(Runtime* runtime,
                           LegateTypeCode code,
                           tuple<size_t> extents,
                           LogicalStore parent, /* = LogicalStore() */
                           std::shared_ptr<TransformStack> transform /*= nullptr*/)
  : impl_(std::make_shared<detail::LogicalStore>(
      runtime, code, std::move(extents), parent.impl_, std::move(transform)))
{
}

LogicalStore::LogicalStore(Runtime* runtime, LegateTypeCode code, const void* data)
  : impl_(std::make_shared<detail::LogicalStore>(runtime, code, data))
{
}

LogicalStore::LogicalStore(std::shared_ptr<detail::LogicalStore> impl) : impl_(std::move(impl)) {}

bool LogicalStore::scalar() const { return impl_->scalar(); }

int32_t LogicalStore::dim() const { return impl_->dim(); }

LegateTypeCode LogicalStore::code() const { return impl_->code(); }

Domain LogicalStore::domain() const { return impl_->domain(); }

const std::vector<size_t>& LogicalStore::extents() const { return impl_->extents(); }

size_t LogicalStore::volume() const { return impl_->volume(); }

std::shared_ptr<LogicalRegionField> LogicalStore::get_storage() { return impl_->get_storage(); }

Legion::Future LogicalStore::get_future() { return impl_->get_future(); }

LogicalStore LogicalStore::promote(int32_t extra_dim, size_t dim_size) const
{
  return LogicalStore(impl_->promote(extra_dim, dim_size, impl_));
}

std::shared_ptr<Store> LogicalStore::get_physical_store(LibraryContext* context)
{
  return impl_->get_physical_store(context);
}

std::unique_ptr<Projection> LogicalStore::find_or_create_partition(const Partition* partition)
{
  return impl_->find_or_create_partition(partition);
}

std::unique_ptr<Partition> LogicalStore::find_or_create_key_partition()
{
  return impl_->find_or_create_key_partition();
}

void LogicalStore::pack(BufferBuilder& buffer) const { impl_->pack(buffer); }

}  // namespace legate
