/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/copy_request.h>

#include <legate/comm/coll.h>
#include <legate/comm/coll_comm.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/detail/copy_util.h>
#include <legate/data/detail/inline_storage.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/traced_exception.h>

#include <nccl.h>
#include <stdexcept>

namespace legate::detail {

void CopyRequest::perform(VariantCode variant_code) const
{
  LEGATE_CHECK(variant_code == VariantCode::CPU);
  LEGATE_CHECK(source_partition_->has_color_shape());

  if (!target_partition_->has_color_shape()) {
    cpu_all_gather_();
    return;
  }

  LEGATE_ABORT("Not implemented yet");
}

namespace {

std::int64_t inner_product(const DomainPoint& rhs1, const DomainPoint& rhs2) noexcept
{
  LEGATE_ASSERT(rhs1.dim == rhs2.dim);

  std::int64_t result = 0;

  for (std::int32_t dim = 0; dim < rhs1.dim; ++dim) {
    result += rhs1[dim] * rhs2[dim];
  }

  return result;
}

std::int64_t calculate_volume(const DomainPoint& shape) noexcept
{
  std::int64_t result = 1;

  for (std::int32_t dim = 0; dim < shape.dim; ++dim) {
    result *= shape[dim];
  }

  return result;
}

DomainPoint calculate_strides(const DomainPoint& shape) noexcept
{
  DomainPoint strides;

  strides.dim = shape.dim;
  // TODO(wonchanl): C order is assumed but the layout can be arbitrary
  strides[strides.dim - 1] = 1;
  for (std::int32_t dim = strides.dim - 2; dim >= 0; --dim) {
    strides[dim] = strides[dim + 1] * shape[dim];
  }

  return strides;
}

DomainPoint to_shape(const Domain& domain) noexcept
{
  DomainPoint shape;

  shape.dim = domain.dim;
  for (std::int32_t dim = 0; dim < shape.dim; ++dim) {
    shape[dim] = domain.rect_data[dim + domain.dim] - domain.rect_data[dim] + 1;
  }

  return shape;
}

template <typename VAL>
void cpu_reorder_elements_(InlineStorage& target, const InlineStorage& source, const Tiling* tiling)
{
  auto* target_ptr       = static_cast<VAL*>(target.data());
  const auto* source_ptr = static_cast<const VAL*>(source.data());

  const auto target_strides = calculate_strides(to_shape(target.domain()));

  const auto& tile_shape  = tiling->tile_shape_dp();
  const auto tile_strides = calculate_strides(tile_shape);
  const auto tile_volume  = calculate_volume(tile_shape);

  const auto& color_strides = calculate_strides(tiling->color_shape_dp());

  for (Domain::DomainPointIterator it{target.domain()}; it; ++it) {
    const auto target_idx = inner_product(it.p, target_strides);
    const auto source_idx = inner_product(it.p % tile_shape, tile_strides) +
                            inner_product(it.p / tile_shape, color_strides) * tile_volume;

    target_ptr[target_idx] = source_ptr[source_idx];
  }
}

}  // namespace

void CopyRequest::cpu_all_gather_() const
{
  // TODO(wonchanl): The code below doesn't handle the case where there are smaller tiles

  // If the source partition is a "batch" partition, the source can be directly all-gathered into
  // the target in the correct order
  //
  // TODO(wonchanl): need to check if the device mesh has C order as well
  if (source_partition_->color_shape().back() == 1) {
    collAllgather(source_->data(),
                  target_->data(),
                  source_->domain().get_volume(),
                  to_coll_data_type_(field_size_),
                  Runtime::get_runtime().get_cpu_comm());
  }

  // Otherwise, the all-gather result needs to be re-ordered into the target

  InlineStorage tmp{target_->domain(), field_size_, target_->target()};

  collAllgather(source_->data(),
                tmp.data(),
                source_->domain().get_volume(),
                to_coll_data_type_(field_size_),
                Runtime::get_runtime().get_cpu_comm());

  // If we're here, the target partition is NoPartition, so the source partition cannot be
  // NoPartition as well.
  const auto* tiling = static_cast<Tiling*>(source_partition_.get());

  switch (field_size_) {
    // FIXME(wonchanl): source_->domain() may not be the tile shape if source_ isn't a tight
    // allocation
    case 1: {
      cpu_reorder_elements_<std::uint8_t>(*target_, tmp, tiling);
      break;
    }
    case 2: {
      cpu_reorder_elements_<std::uint16_t>(*target_, tmp, tiling);
      break;
    }
    case 4: {
      cpu_reorder_elements_<std::uint32_t>(*target_, tmp, tiling);
      break;
    }
    case 8: {
      cpu_reorder_elements_<std::uint64_t>(*target_, tmp, tiling);
      break;
    }
    default: {  // legate-lint: no-switch-default
      throw TracedException<std::runtime_error>{"Unsupported field size"};
    }
  }
}

}  // namespace legate::detail
