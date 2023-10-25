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

#include "core/mapping/mapping.h"

namespace legate::mapping {

inline const detail::DimOrdering* DimOrdering::impl() const noexcept { return impl_.get(); }

inline DimOrdering::DimOrdering(std::shared_ptr<detail::DimOrdering> impl) : impl_{std::move(impl)}
{
}

inline const detail::StoreMapping* StoreMapping::impl() const noexcept { return impl_.get(); }

inline detail::StoreMapping* StoreMapping::release() noexcept { return impl_.release(); }

}  // namespace legate::mapping
