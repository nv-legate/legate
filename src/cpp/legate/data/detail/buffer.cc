/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/buffer.h>

#include <legate/data/detail/raw_allocation.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace legate::detail {

Buffer::Buffer(InternalSharedPtr<RawAllocation> allocation,
               InternalSharedPtr<Type> type,
               Rect domain,
               SmallVector<std::int32_t, LEGATE_MAX_DIM> dim_order)
  : allocation_{std::move(allocation)},
    type_{std::move(type)},
    domain_{std::move(domain)},
    dim_order_{std::move(dim_order)}
{
  if (dim_order_.empty()) {
    // No ordering given: default to C order, i.e. the last dimension is the unit-stride one.
    dim_order_ =
      mapping::detail::DimOrdering{mapping::detail::DimOrdering::Kind::C}.dimension_order(
        domain_.dim());
  } else if (dim_order_.size() != domain_.dim()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Dimension ordering has {} entries but the buffer domain {} is {}-D",
                  dim_order_.size(),
                  domain_,
                  domain_.dim())};
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    if (allocation_->size() < domain_.volume() * type_->size()) {
      throw TracedException<std::invalid_argument>{
        fmt::format("Raw allocation is not big enough for the domain (allocation size: {}, domain: "
                    "{}, element size: {})",
                    allocation_->size(),
                    domain_.volume(),
                    type_->size())};
    }
  }
}

mapping::StoreTarget Buffer::target() const { return allocation()->target(); }

void* Buffer::data() { return allocation()->ptr(); }

const void* Buffer::data() const { return allocation()->ptr(); }

Point Buffer::element_strides() const
{
  auto strides        = Point::zeroes(static_cast<std::int32_t>(domain_.dim()));
  auto&& extents      = domain_.extents();
  std::int64_t stride = 1;

  // `dim_order_` runs from the most rapidly changing dimension to the least rapidly changing one,
  // so walking it in order accumulates the strides of a dense layout.
  for (auto&& dim : dim_order_) {
    strides[dim] = stride;
    stride *= extents[dim];
  }
  return strides;
}

std::pair<void*, Point> Buffer::base_and_element_strides() const
{
  auto strides = element_strides();
  auto* const base =
    static_cast<std::int8_t*>(allocation()->ptr()) - (domain_.lo().dot(strides) * type_->size());

  return {static_cast<void*>(base), std::move(strides)};
}

/*static*/ InternalSharedPtr<Buffer> Buffer::allocate(InternalSharedPtr<Type> type,
                                                      Rect domain,
                                                      mapping::StoreTarget target)
{
  auto allocation =
    RawAllocation::create(target, std::max<>(domain.volume(), std::size_t{1}) * type->size());

  return make_internal_shared<Buffer>(std::move(allocation), std::move(type), std::move(domain));
}

}  // namespace legate::detail
