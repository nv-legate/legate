/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/buffer.h>

#include <legate/data/detail/raw_allocation.h>
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
               Rect domain)
  : allocation_{std::move(allocation)}, type_{std::move(type)}, domain_{std::move(domain)}
{
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

std::pair<void*, Point> Buffer::base_and_element_strides() const
{
  auto element_strides = domain_.strides();
  auto* const base     = static_cast<std::int8_t*>(allocation()->ptr()) -
                         (domain_.lo().dot(element_strides) * type_->size());

  return {static_cast<void*>(base), std::move(element_strides)};
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
