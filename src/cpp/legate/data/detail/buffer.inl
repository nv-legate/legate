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

inline void Buffer::reshape(Rect new_domain) { domain_ = new_domain; }

}  // namespace legate::detail
