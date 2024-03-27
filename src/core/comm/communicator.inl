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

#include "core/comm/communicator.h"

#include <utility>

namespace legate::comm {

inline Communicator::Communicator(Legion::Future future) : future_{std::move(future)} {}

template <typename T>
T Communicator::get() const
{
  return future_.get_result<T>();
}

}  // namespace legate::comm
