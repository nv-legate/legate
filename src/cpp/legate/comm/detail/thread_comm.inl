/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/detail/thread_comm.h>

namespace legate::detail::comm::coll {

inline bool ThreadComm::ready() const { return ready_flag_; }

inline const ThreadComm::atomic_buffer_type* ThreadComm::buffers() const { return buffers_.get(); }

inline ThreadComm::atomic_buffer_type* ThreadComm::buffers() { return buffers_.get(); }

inline const ThreadComm::atomic_displ_type* ThreadComm::displs() const { return displs_.get(); }

inline ThreadComm::atomic_displ_type* ThreadComm::displs() { return displs_.get(); }

}  // namespace legate::detail::comm::coll
