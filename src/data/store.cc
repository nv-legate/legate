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

#include "data/store.h"
#include "utilities/dispatch.h"

namespace legate {

using namespace Legion;

RegionField::RegionField(int32_t dim, const PhysicalRegion& pr, FieldID fid, unsigned reqIdx)
  : dim_(dim), pr_(pr), fid_(fid), reqIdx_(reqIdx)
{
  auto priv  = pr.get_privilege();
  readable_  = static_cast<bool>(priv & LEGION_READ_PRIV);
  writable_  = static_cast<bool>(priv & LEGION_WRITE_PRIV);
  reducible_ = static_cast<bool>(priv & LEGION_REDUCE) || (readable_ && writable_);
}

RegionField::RegionField(RegionField&& other) noexcept
  : dim_(other.dim_),
    pr_(other.pr_),
    fid_(other.fid_),
    reqIdx_(other.reqIdx_),
    readable_(other.readable_),
    writable_(other.writable_),
    reducible_(other.reducible_)
{
}

RegionField& RegionField::operator=(RegionField&& other) noexcept
{
  dim_       = other.dim_;
  pr_        = other.pr_;
  fid_       = other.fid_;
  reqIdx_    = other.reqIdx_; 

  readable_  = other.readable_;
  writable_  = other.writable_;
  reducible_ = other.reducible_;
  return *this;
}

Domain RegionField::domain() const { return dim_dispatch(dim_, get_domain_fn{}, pr_); }

OutputRegionField::OutputRegionField(const OutputRegion& out, FieldID fid, unsigned reqIdx) : out_(out), fid_(fid), reqIdx_(reqIdx) {}

OutputRegionField::OutputRegionField(OutputRegionField&& other) noexcept
  : bound_(other.bound_), out_(other.out_), fid_(other.fid_), reqIdx_(other.reqIdx_)
{
  other.bound_ = false;
  other.out_   = OutputRegion();
  other.fid_   = -1;
  //TODO, how should we invalidate reqIdx
}

OutputRegionField& OutputRegionField::operator=(OutputRegionField&& other) noexcept
{
  bound_ = other.bound_;
  out_   = other.out_;
  fid_   = other.fid_;
  reqIdx_= other.reqIdx_;

  other.bound_ = false;
  other.out_   = OutputRegion();
  other.fid_   = -1;
  //TODO, how should we invalidate reqIdx

  return *this;
}

FutureWrapper::FutureWrapper(Domain domain, Future future) : domain_(domain), future_(future) {}

FutureWrapper::FutureWrapper(const FutureWrapper& other) noexcept
  : domain_(other.domain_), future_(other.future_)
{
}

FutureWrapper& FutureWrapper::operator=(const FutureWrapper& other) noexcept
{
  domain_ = other.domain_;
  future_ = other.future_;
  return *this;
}

Domain FutureWrapper::domain() const { return domain_; }

Store::Store(int32_t dim,
             LegateTypeCode code,
             FutureWrapper future,
             std::unique_ptr<StoreTransform> transform)
  : is_future_(true),
    is_output_store_(false),
    dim_(dim),
    code_(code),
    redop_id_(-1),
    future_(future),
    transform_(std::move(transform)),
    readable_(true)
{
}

Store::Store(int32_t dim,
             LegateTypeCode code,
             int32_t redop_id,
             RegionField&& region_field,
             std::unique_ptr<StoreTransform> transform)
  : is_future_(false),
    is_output_store_(false),
    dim_(dim),
    code_(code),
    redop_id_(redop_id),
    region_field_(std::forward<RegionField>(region_field)),
    transform_(std::move(transform))
{
  readable_  = region_field_.is_readable();
  writable_  = region_field_.is_writable();
  reducible_ = region_field_.is_reducible();
}

Store::Store(LegateTypeCode code,
             OutputRegionField&& output,
             std::unique_ptr<StoreTransform> transform)
  : is_future_(false),
    is_output_store_(true),
    dim_(-1),
    code_(code),
    redop_id_(-1),
    output_field_(std::forward<OutputRegionField>(output)),
    transform_(std::move(transform))
{
}

Store::Store(Store&& other) noexcept
  : is_future_(other.is_future_),
    is_output_store_(other.is_output_store_),
    dim_(other.dim_),
    code_(other.code_),
    redop_id_(other.redop_id_),
    future_(other.future_),
    region_field_(std::forward<RegionField>(other.region_field_)),
    output_field_(std::forward<OutputRegionField>(other.output_field_)),
    transform_(std::move(other.transform_)),
    readable_(other.readable_),
    writable_(other.writable_),
    reducible_(other.reducible_)
{
}

Store& Store::operator=(Store&& other) noexcept
{
  is_future_       = other.is_future_;
  is_output_store_ = other.is_output_store_;
  dim_             = other.dim_;
  code_            = other.code_;
  redop_id_        = other.redop_id_;
  if (is_future_)
    future_ = other.future_;
  else if (is_output_store_)
    output_field_ = std::move(other.output_field_);
  else
    region_field_ = std::move(other.region_field_);
  transform_ = std::move(other.transform_);
  readable_  = other.readable_;
  writable_  = other.writable_;
  reducible_ = other.reducible_;
  return *this;
}

Domain Store::domain() const
{
  assert(!is_output_store_);
  auto result = is_future_ ? future_.domain() : region_field_.domain();
  if (nullptr != transform_) result = transform_->transform(result);
  assert(result.dim == dim_);
  return result;
}

}  // namespace legate
