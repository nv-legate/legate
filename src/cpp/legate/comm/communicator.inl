/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/communicator.h>

#include <utility>

namespace legate::comm {

inline Communicator::Communicator(void* comm) : comm_{comm} {}

template <typename T>
T Communicator::get() const
{
  return *static_cast<const T*>(comm_);
}

}  // namespace legate::comm
