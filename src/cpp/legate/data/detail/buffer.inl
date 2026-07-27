/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/buffer.h>

namespace legate::detail {

inline const InternalSharedPtr<Type>& Buffer::type() const { return type_; }

inline std::uint32_t Buffer::dim() const { return domain().dim(); }

inline const Rect& Buffer::domain() const { return domain_; }

inline const InternalSharedPtr<RawAllocation>& Buffer::allocation() const { return allocation_; }

inline Span<const std::int32_t> Buffer::dim_order() const
{
  return Span<const std::int32_t>{dim_order_.data(), dim_order_.size()};
}

inline void Buffer::reshape(Rect new_domain)
{
  // Reshaping only re-bases/resizes the domain; the dimension ordering (and hence which dimension
  // is the unit-stride one) is unchanged, so it must keep the same dimensionality.
  LEGATE_ASSERT(new_domain.dim() == domain_.dim());
  domain_ = new_domain;
}

}  // namespace legate::detail
