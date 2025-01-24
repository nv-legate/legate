/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <legate/comm/coll_comm.h>
#include <legate/utilities/typedefs.h>

namespace legate::comm::coll {

// NOLINTBEGIN(readability-identifier-naming)
void collCommCreate(CollComm global_comm,
                    int global_comm_size,
                    int global_rank,
                    int unique_id,
                    const int* mapping_table);

void collCommDestroy(CollComm global_comm);

void collAlltoallv(const void* sendbuf,
                   const int sendcounts[],
                   const int sdispls[],
                   void* recvbuf,
                   const int recvcounts[],
                   const int rdispls[],
                   CollDataType type,
                   CollComm global_comm);

void collAlltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

void collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);
// NOLINTEND(readability-identifier-naming)

}  // namespace legate::comm::coll
