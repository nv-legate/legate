/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/future_wrapper.h>
#include <legate/utilities/assert.h>

namespace legate::detail {

inline std::int32_t FutureWrapper::dim() const { return domain().dim; }

inline const Domain& FutureWrapper::domain() const { return domain_; }

inline bool FutureWrapper::valid() const { return get_future().valid(); }

inline std::uint32_t FutureWrapper::field_size() const { return field_size_; }

inline std::size_t FutureWrapper::field_offset() const { return field_offset_; }

inline bool FutureWrapper::is_read_only() const { return read_only_; }

inline const Legion::Future& FutureWrapper::get_future() const { return future_; }

inline const Legion::UntypedDeferredValue& FutureWrapper::get_buffer() const
{
  LEGATE_ASSERT(!is_read_only());
  return buffer_;
}

}  // namespace legate::detail
