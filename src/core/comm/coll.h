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

#include "core/comm/coll_comm.h"
#include "core/utilities/typedefs.h"

namespace legate::comm::coll {

namespace detail {

[[nodiscard]] Logger& log_coll();

}  // namespace detail

// NOLINTBEGIN(readability-identifier-naming)
[[nodiscard]] CollStatus collCommCreate(CollComm global_comm,
                                        int global_comm_size,
                                        int global_rank,
                                        int unique_id,
                                        const int* mapping_table);

[[nodiscard]] CollStatus collCommDestroy(CollComm global_comm);

[[nodiscard]] CollStatus collAlltoallv(const void* sendbuf,
                                       const int sendcounts[],
                                       const int sdispls[],
                                       void* recvbuf,
                                       const int recvcounts[],
                                       const int rdispls[],
                                       CollDataType type,
                                       CollComm global_comm);

[[nodiscard]] CollStatus collAlltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

[[nodiscard]] CollStatus collAllgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm);

[[nodiscard]] CollStatus collInit(int argc, char* argv[]);

[[nodiscard]] CollStatus collFinalize();

// this is forward declared in src/core/utilities/abort.h (for LEGATE_ABORT()), because we don't
// want to include this entire header
void collAbort() noexcept;  // NOLINT(readability-redundant-declaration)

[[nodiscard]] int collInitComm();
// NOLINTEND(readability-identifier-naming)

}  // namespace legate::comm::coll
