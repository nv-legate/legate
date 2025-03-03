/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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
