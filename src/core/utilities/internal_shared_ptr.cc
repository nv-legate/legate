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

#include "core/utilities/internal_shared_ptr.h"

namespace legate {

BadInternalWeakPtr::BadInternalWeakPtr(std::string what) noexcept : what_{std::move(what)} {}

const char* BadInternalWeakPtr::what() const noexcept { return what_.c_str(); }

}  // namespace legate
