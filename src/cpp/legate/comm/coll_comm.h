/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/macros.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail::comm::coll {
class ThreadComm;
}  // namespace legate::detail::comm::coll

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

enum class CollCommType : std::uint8_t { CollMPI, CollLocal };

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
  std::ptrdiff_t mpi_comm{};
  RankMappingTable mapping_table{};
#endif
  detail::comm::coll::ThreadComm* local_comm{};
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

}  // namespace legate::comm::coll
