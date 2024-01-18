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

#include "core/data/shape.h"

#include "core/data/detail/shape.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace legate {

Shape::Shape(tuple<uint64_t> extents)
  : impl_{make_internal_shared<detail::Shape>(std::move(extents))}
{
}

const tuple<uint64_t>& Shape::extents() const { return impl_->extents(); }

uint32_t Shape::ndim() const { return impl_->ndim(); }

std::string Shape::to_string() const { return impl_->to_string(); }

bool Shape::operator==(const Shape& other) const { return *impl_ == *other.impl_; }

}  // namespace legate
