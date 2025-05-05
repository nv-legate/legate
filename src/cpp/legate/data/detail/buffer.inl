/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/buffer.h>

namespace legate::detail {

inline const InternalSharedPtr<Type>& TaskLocalBuffer::type() const { return type_; }

inline std::int32_t TaskLocalBuffer::dim() const { return domain().get_dim(); }

inline const Domain& TaskLocalBuffer::domain() const { return domain_; }

inline const Legion::UntypedDeferredBuffer<>& TaskLocalBuffer::legion_buffer() const
{
  return buf_;
}

}  // namespace legate::detail
