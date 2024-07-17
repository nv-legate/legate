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

#include "core/comm/detail/thread_comm.h"

namespace legate::detail::comm::coll {

inline bool ThreadComm::ready() const { return ready_flag_; }

inline const ThreadComm::atomic_buffer_type* ThreadComm::buffers() const { return buffers_.get(); }

inline ThreadComm::atomic_buffer_type* ThreadComm::buffers() { return buffers_.get(); }

inline const ThreadComm::atomic_displ_type* ThreadComm::displs() const { return displs_.get(); }

inline ThreadComm::atomic_displ_type* ThreadComm::displs() { return displs_.get(); }

}  // namespace legate::detail::comm::coll
