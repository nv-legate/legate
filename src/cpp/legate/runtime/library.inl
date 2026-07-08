/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate_defines.h>

#include <legate/runtime/library.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

namespace legate {

inline Library::Library(detail::Library* impl) : impl_{impl} {}

template <typename REDOP>
GlobalRedopID Library::register_reduction_operator(LocalRedopID)
{
  // TODO(wonchanl): Need custom reduction op support in nvshard
  return GlobalRedopID{0};
}

inline bool Library::operator==(const Library& other) const { return impl() == other.impl(); }

inline bool Library::operator!=(const Library& other) const { return !(*this == other); }

inline const detail::Library* Library::impl() const { return impl_; }

inline detail::Library* Library::impl() { return impl_; }

}  // namespace legate
