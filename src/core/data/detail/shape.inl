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

namespace legate::detail {

inline Shape::Shape(uint32_t dim) : dim_{dim} {}

inline bool Shape::ready() const { return state_ == State::READY; }

inline uint32_t Shape::ndim() const { return dim_; }

inline size_t Shape::volume() { return extents().volume(); }

inline bool Shape::operator!=(Shape& other) { return !operator==(other); }

}  // namespace legate::detail
