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

#include "data/transform.h"

namespace legate {

using namespace Legion;

using StoreTransformP = std::shared_ptr<StoreTransform>;

/*
typedef enum legate_core_transform_t {
  LEGATE_CORE_TRANSFORM_SHIFT = 100,
  LEGATE_CORE_TRANSFORM_PROMOTE,
  LEGATE_CORE_TRANSFORM_PROJECT,
  LEGATE_CORE_TRANSFORM_TRANSPOSE,
  LEGATE_CORE_TRANSFORM_DELINEARIZE,
} legate_core_transform_t;
*/


DomainAffineTransform combine(const DomainAffineTransform& lhs, const DomainAffineTransform& rhs)
{
  DomainAffineTransform result;
  auto transform   = lhs.transform * rhs.transform;
  auto offset      = lhs.transform * rhs.offset + lhs.offset;
  result.transform = transform;
  result.offset    = offset;
  return result;
}

StoreTransform::StoreTransform(StoreTransformP parent) : parent_(std::move(parent)) {}

Shift::Shift(int32_t dim, int64_t offset, StoreTransformP parent)
  : StoreTransform(std::forward<StoreTransformP>(parent)), dim_(dim), offset_(offset)
{
}

int32_t Shift::getTransformCode() const
{
    return LEGATE_CORE_TRANSFORM_SHIFT;
}

Domain Shift::transform(const Domain& input) const
{
  auto result = nullptr != parent_ ? parent_->transform(input) : input;
  result.rect_data[dim_] += offset_;
  result.rect_data[dim_ + result.dim] += offset_;
  return result;
}

DomainAffineTransform Shift::inverse_transform(int32_t in_dim) const
{
  assert(dim_ < in_dim);
  auto out_dim = in_dim;

  DomainTransform transform;
  transform.m = out_dim;
  transform.n = in_dim;
  for (int32_t i = 0; i < out_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j)
      transform.matrix[i * in_dim + j] = static_cast<coord_t>(i == j);

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = i == dim_ ? -offset_ : 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  } else
    return result;
}

Promote::Promote(int32_t extra_dim, int64_t dim_size, StoreTransformP parent)
  : StoreTransform(std::forward<StoreTransformP>(parent)),
    extra_dim_(extra_dim),
    dim_size_(dim_size)
{
}

int32_t Promote::getTransformCode() const
{
    return LEGATE_CORE_TRANSFORM_PROMOTE;
}

Domain Promote::transform(const Domain& input) const
{
  auto promote = [](int32_t extra_dim, int64_t dim_size, const Domain& input) {
    Domain output;
    output.dim = input.dim + 1;

    for (int32_t out_dim = 0, in_dim = 0; out_dim < output.dim; ++out_dim)
      if (out_dim == extra_dim) {
        output.rect_data[out_dim]              = 0;
        output.rect_data[out_dim + output.dim] = dim_size - 1;
      } else {
        output.rect_data[out_dim]              = input.rect_data[in_dim];
        output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
        ++in_dim;
      }
    return output;
  };

  return promote(extra_dim_, dim_size_, nullptr != parent_ ? parent_->transform(input) : input);
}

DomainAffineTransform Promote::inverse_transform(int32_t in_dim) const
{
  assert(extra_dim_ < in_dim);
  auto out_dim = in_dim - 1;

  DomainTransform transform;
  transform.m = out_dim;
  transform.n = in_dim;
  for (int32_t i = 0; i < out_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

  for (int32_t j = 0, i = 0; j < in_dim; ++j)
    if (j != extra_dim_) transform.matrix[i++ * in_dim + j] = 1;

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  } else
    return result;
}

Project::Project(int32_t dim, int64_t coord, StoreTransformP parent)
  : StoreTransform(std::forward<StoreTransformP>(parent)), dim_(dim), coord_(coord)
{
}

int32_t Project::getTransformCode() const
{
    return LEGATE_CORE_TRANSFORM_PROJECT;
}

Domain Project::transform(const Domain& input) const
{
  auto project = [](int32_t collapsed_dim, const Domain& input) {
    Domain output;
    output.dim = input.dim - 1;

    for (int32_t in_dim = 0, out_dim = 0; in_dim < input.dim; ++in_dim)
      if (in_dim != collapsed_dim) {
        output.rect_data[out_dim]              = input.rect_data[in_dim];
        output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
        ++out_dim;
      }
    return output;
  };

  return project(dim_, nullptr != parent_ ? parent_->transform(input) : input);
}

DomainAffineTransform Project::inverse_transform(int32_t in_dim) const
{
  auto out_dim = in_dim + 1;
  assert(dim_ < out_dim);

  DomainTransform transform;
  transform.m = out_dim;
  if (in_dim == 0) {
    transform.n         = out_dim;
    transform.matrix[0] = 0;
  } else {
    transform.n = in_dim;
    for (int32_t i = 0; i < out_dim; ++i)
      for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

    for (int32_t i = 0, j = 0; i < out_dim; ++i)
      if (i != dim_) transform.matrix[i * in_dim + j++] = 1;
  }

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = i == dim_ ? coord_ : 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  } else
    return result;
}

Transpose::Transpose(std::vector<int32_t>&& axes, StoreTransformP parent)
  : StoreTransform(std::forward<StoreTransformP>(parent)), axes_(std::move(axes))
{
}

int32_t Transpose::getTransformCode() const
{
    return LEGATE_CORE_TRANSFORM_TRANSPOSE;
}

Domain Transpose::transform(const Domain& input) const
{
  auto transpose = [](const auto& axes, const Domain& input) {
    Domain output;
    output.dim = input.dim;
    for (int32_t in_dim = 0; in_dim < input.dim; ++in_dim) {
      auto out_dim                           = axes[in_dim];
      output.rect_data[out_dim]              = input.rect_data[in_dim];
      output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
    }
    return output;
  };

  return transpose(axes_, nullptr != parent_ ? parent_->transform(input) : input);
}

DomainAffineTransform Transpose::inverse_transform(int32_t in_dim) const
{
  DomainTransform transform;
  transform.m = in_dim;
  transform.n = in_dim;
  for (int32_t i = 0; i < in_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

  for (int32_t i = 0, j = 0; i < in_dim; ++i) transform.matrix[i * in_dim + axes_[i]] = 1;

  DomainPoint offset;
  offset.dim = in_dim;
  for (int32_t i = 0; i < in_dim; ++i) offset[i] = 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(in_dim);
    return combine(parent, result);
  } else
    return result;
}

Delinearize::Delinearize(int32_t dim, std::vector<int64_t>&& sizes, StoreTransformP parent)
  : StoreTransform(std::forward<StoreTransformP>(parent)),
    dim_(dim),
    sizes_(std::move(sizes)),
    strides_(sizes_.size(), 1),
    volume_(1)
{
  for (int32_t dim = sizes_.size() - 2; dim >= 0; --dim)
    strides_[dim] = strides_[dim + 1] * sizes_[dim + 1];
  for (auto size : sizes_) volume_ *= size;
}

int32_t Delinearize::getTransformCode() const
{
    return LEGATE_CORE_TRANSFORM_DELINEARIZE;
}

Domain Delinearize::transform(const Domain& input) const
{
  auto delinearize = [](const auto dim, const auto ndim, const auto& strides, const Domain& input) {
    Domain output;
    output.dim     = input.dim - 1 + ndim;
    int32_t in_dim = 0;
    for (int32_t in_dim = 0, out_dim = 0; in_dim < input.dim; ++in_dim) {
      if (in_dim == dim) {
        auto lo = input.rect_data[in_dim];
        auto hi = input.rect_data[input.dim + in_dim];
        for (auto stride : strides) {
          output.rect_data[out_dim]              = lo / stride;
          output.rect_data[output.dim + out_dim] = hi / stride;
          lo                                     = lo % stride;
          hi                                     = hi % stride;
          ++out_dim;
        }
      } else {
        output.rect_data[out_dim]              = input.rect_data[in_dim];
        output.rect_data[output.dim + out_dim] = input.rect_data[input.dim + in_dim];
        ++out_dim;
      }
    }
    return output;
  };
  return delinearize(
    dim_, sizes_.size(), strides_, nullptr != parent_ ? parent_->transform(input) : input);
}

DomainAffineTransform Delinearize::inverse_transform(int32_t in_dim) const
{
  DomainTransform transform;
  int32_t out_dim = in_dim - strides_.size() + 1;
  transform.m     = out_dim;
  transform.n     = in_dim;
  for (int32_t i = 0; i < out_dim; ++i)
    for (int32_t j = 0; j < in_dim; ++j) transform.matrix[i * in_dim + j] = 0;

  for (int32_t i = 0, j = 0; i < out_dim; ++i)
    if (i == dim_)
      for (auto stride : strides_) transform.matrix[i * in_dim + j++] = stride;
    else
      transform.matrix[i * in_dim + j++] = 1;

  DomainPoint offset;
  offset.dim = out_dim;
  for (int32_t i = 0; i < out_dim; ++i) offset[i] = 0;

  DomainAffineTransform result;
  result.transform = transform;
  result.offset    = offset;

  if (nullptr != parent_) {
    auto parent = parent_->inverse_transform(out_dim);
    return combine(parent, result);
  } else
    return result;
}

}  // namespace legate
