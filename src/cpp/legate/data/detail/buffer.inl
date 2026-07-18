/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/buffer.h>

namespace legate::detail {

inline const InternalSharedPtr<Type>& TaskLocalBuffer::type() const { return type_; }

inline std::uint32_t TaskLocalBuffer::dim() const { return domain().dim(); }

inline const Domain& TaskLocalBuffer::domain() const { return domain_; }

inline const InternalSharedPtr<InlineStorage>& TaskLocalBuffer::get_storage() const
{
  return storage_;
}

}  // namespace legate::detail
