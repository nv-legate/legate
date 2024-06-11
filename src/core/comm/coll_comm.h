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

#include "core/comm/thread_comm.h"
#include "core/utilities/macros.h"

#include <cstdint>

// TODO(jfaibussowit)
// Decouple these
#if LEGATE_DEFINED(LEGATE_USE_NETWORK)
#include <mpi.h>
#endif

namespace legate::comm::coll {

// NOLINTBEGIN(readability-identifier-naming)
enum class CollDataType : std::uint8_t {
  CollInt8,
  CollChar,
  CollUint8,
  CollInt,
  CollUint32,
  CollInt64,
  CollUint64,
  CollFloat,
  CollDouble,
};

enum CollStatus : std::uint8_t {
  CollSuccess,
  CollError,
};

enum CollCommType : std::uint8_t {
  CollMPI,
  CollLocal,
};

// TODO(jfaibussowit)
// Decouple these
#if LEGATE_DEFINED(LEGATE_USE_NETWORK)
class RankMappingTable {
 public:
  int* mpi_rank{};
  int* global_rank{};
};
#endif

class Coll_Comm {
 public:
  // TODO(jfaibussowit)
  // Decouple these
#if LEGATE_DEFINED(LEGATE_USE_NETWORK)
  MPI_Comm mpi_comm{};
  RankMappingTable mapping_table{};
#endif
  ThreadComm* local_comm{};
  int mpi_rank{};
  int mpi_comm_size{};
  int mpi_comm_size_actual{};
  int global_rank{};
  int global_comm_size{};
  int nb_threads{};
  int unique_id{};
  bool status{};
};
// NOLINTEND(readability-identifier-naming)

using CollComm = Coll_Comm*;

#define LEGATE_CHECK_COLL(...)                                                           \
  do {                                                                                   \
    const auto ret = __VA_ARGS__;                                                        \
    if (LEGATE_UNLIKELY(ret != legate::comm::coll::CollStatus::CollSuccess)) return ret; \
  } while (0)

}  // namespace legate::comm::coll
