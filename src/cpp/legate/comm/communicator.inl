/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/communicator.h>

#include <utility>

namespace legate::comm {

inline Communicator::Communicator(Legion::Future future) : future_{std::move(future)} {}

template <typename T>
T Communicator::get() const
{
  return future_.get_result<T>();
}

}  // namespace legate::comm
