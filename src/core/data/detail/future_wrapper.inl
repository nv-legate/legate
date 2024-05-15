/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/data/detail/future_wrapper.h"

namespace legate::detail {

inline std::int32_t FutureWrapper::dim() const { return domain().dim; }

inline const Domain& FutureWrapper::domain() const { return domain_; }

inline bool FutureWrapper::valid() const { return get_future().valid(); }

inline std::uint32_t FutureWrapper::field_size() const { return field_size_; }

inline bool FutureWrapper::is_read_only() const { return read_only_; }

inline const Legion::Future& FutureWrapper::get_future() const { return future_; }

inline const Legion::UntypedDeferredValue& FutureWrapper::get_buffer() const { return buffer_; }

}  // namespace legate::detail
