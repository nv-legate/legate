/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll_comm.h>
#include <legate/utilities/typedefs.h>

namespace legate::comm::coll {

// NOLINTBEGIN(readability-identifier-naming)
LEGATE_EXPORT void collCommCreate(CollComm global_comm,
                                  int global_comm_size,
                                  int global_rank,
                                  int unique_id,
                                  const int* mapping_table);

LEGATE_EXPORT void collCommDestroy(CollComm global_comm);

LEGATE_EXPORT void collAlltoallv(const void* sendbuf,
                                 const int sendcounts[],
                                 const int sdispls[],
                                 void* recvbuf,
                                 const int recvcounts[],
                                 const int rdispls[],
                                 CollDataType type,
                                 CollComm global_comm);

LEGATE_EXPORT void collAlltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

LEGATE_EXPORT void collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

/**
 * @brief Perform an all-reduce operation among the ranks of the global communicator.
 * Bitwise and logical operations are not supported for floating point types.
 *
 * @param sendbuf  The source buffer to reduce. This buffer must be of size count x CollDataType
 * size.
 * @param recvbuf The destination buffer to receive the reduced result into. This buffer must be of
 * size count x CollDataType size.
 * @param count The number of elements to reduce.
 * @param type The data type of the elements.
 * @param op The reduction operation to perform.
 * @param global_comm The global communicator.
 *
 * @throw std::invalid_argument if the reduction operation is not supported for the data type.
 */
LEGATE_EXPORT void collAllreduce(const void* sendbuf,
                                 void* recvbuf,
                                 int count,
                                 CollDataType type,
                                 ReductionOpKind op,
                                 CollComm global_comm);
// NOLINTEND(readability-identifier-naming)

}  // namespace legate::comm::coll
