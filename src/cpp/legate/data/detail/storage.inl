/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/shape.h>
#include <legate/data/detail/storage.h>

namespace legate::detail {

inline std::uint64_t Storage::id() const { return storage_id_; }

inline bool Storage::replicated() const { return replicated_; }

inline bool Storage::unbound() const { return unbound_; }

inline const InternalSharedPtr<Shape>& Storage::shape() const { return shape_; }

inline Span<const std::uint64_t> Storage::extents() const { return shape()->extents(); }

inline std::size_t Storage::volume() const { return shape()->volume(); }

inline std::uint32_t Storage::dim() const { return shape()->ndim(); }

inline std::int32_t Storage::level() const { return level_; }

inline std::size_t Storage::scalar_offset() const { return scalar_offset_; }

inline std::string_view Storage::provenance() const { return provenance_; }

}  // namespace legate::detail
