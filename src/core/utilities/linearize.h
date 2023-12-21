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

#include "core/utilities/typedefs.h"

namespace legate {

[[nodiscard]] size_t linearize(const DomainPoint& lo,
                               const DomainPoint& hi,
                               const DomainPoint& point);

[[nodiscard]] DomainPoint delinearize(const DomainPoint& lo, const DomainPoint& hi, size_t idx);

}  // namespace legate
