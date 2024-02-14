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

#include "core/utilities/detail/malloc.h"
#include "core/utilities/detail/scope_guard.h"
#include "core/utilities/typedefs.h"

#include "coll.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace legate::comm::coll {

enum CollTag : int {
  BCAST_TAG     = 0,
  GATHER_TAG    = 1,
  ALLTOALL_TAG  = 2,
  ALLTOALLV_TAG = 3,
  MAX_TAG       = 10,
};

#define CHECK_COLL(...)                 \
  do {                                  \
    const auto ret = __VA_ARGS__;       \
    if (ret != CollSuccess) return ret; \
  } while (0)

namespace {

[[nodiscard]] std::pair<int, int> mostFrequent(const int* arr, int n)
{
  std::unordered_map<int, int> hash;
  for (int i = 0; i < n; i++) {
    hash[arr[i]]++;
  }

  // find the max frequency
  int max_count = 0;
  for (auto&& [_, count] : hash) {
    max_count = std::max(count, max_count);
  }

  return {max_count, hash.size()};
}

[[nodiscard]] int match2ranks(int rank1, int rank2, CollComm global_comm)
{
  // tag: seg idx + rank_idx + tag
  // send_tag = sendto_global_rank * 10000 + global_rank (concat 2 ranks)
  // which dst seg it sends to (in dst rank)
  // recv_tag = global_rank * 10000 + recvfrom_global_rank (concat 2 ranks)
  // idx of current seg we are receving (in src/my rank)
  // example:
  // 00 | 01 | 02 | 03
  // 10 | 11 | 12 | 13
  // 20 | 21 | 22 | 23
  // 30 | 31 | 32 | 33
  // 01's send_tag = 10, 10's recv_tag = 10, match
  // 12's send_tag = 21, 21's recv_tag = 21, match

  int tag;
  // old tagging system for debug
  // constexpr int const max_ranks = 10000;
  // tag                           = rank1 * max_ranks + rank2;

  // new tagging system, if crash, switch to the old one

  tag = rank1 % global_comm->nb_threads * global_comm->global_comm_size + rank2;

  // Szudzik's Function, two numbers < 32768
  // if (rank1 >= rank2) {
  //   tag = rank1*rank1 + rank1 + rank2;
  // } else {
  //   tag = rank1 + rank2*rank2;
  // }

  // Cantor Pairing Function, two numbers < 32768
  // tag = (rank1 + rank2) * (rank1 + rank2 + 1) / 2 + rank1;

  return tag;
}

void check_mpi(int error, const char* file, int line, const char* func)
{
  if (error == MPI_SUCCESS) {
    return;
  }
  int init = 0;

  static_cast<void>(fprintf(
    stderr, "Internal MPI failure with error code %d in %s:%d in %s()\n", error, file, line, func));
  static_cast<void>(MPI_Initialized(&init));
  if (init) {
    static_cast<void>(MPI_Abort(MPI_COMM_WORLD, error));
  }
  std::abort();
}

}  // namespace

#define CHECK_MPI(...)                                                     \
  do {                                                                     \
    const int result = __VA_ARGS__;                                        \
    ::legate::comm::coll::check_mpi(result, __FILE__, __LINE__, __func__); \
  } while (false)

// public functions start from here

MPINetwork::MPINetwork(int /*argc*/, char* /*argv*/[])
{
  detail::log_coll().debug("Enable MPINetwork");
  LegateCheck(current_unique_id == 0);
  int provided, init_flag = 0;
  CHECK_MPI(MPI_Initialized(&init_flag));
  if (!init_flag) {
    detail::log_coll().info("MPI being initialized by legate");
    CHECK_MPI(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
    self_init_mpi = true;
  }
  int mpi_thread_model;
  CHECK_MPI(MPI_Query_thread(&mpi_thread_model));
  if (mpi_thread_model != MPI_THREAD_MULTIPLE) {
    LEGATE_ABORT(
      "MPI has been initialized by others, but is not initialized with "
      "MPI_THREAD_MULTIPLE");
  }
  // check
  int *tag_ub, flag;
  CHECK_MPI(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag));
  LegateCheck(flag);
  mpi_tag_ub = *tag_ub;
  LegateCheck(mpi_comms.empty());
  BackendNetwork::coll_inited = true;
  BackendNetwork::comm_type   = CollCommType::CollMPI;
}

MPINetwork::~MPINetwork()
{
  detail::log_coll().debug("Finalize MPINetwork");
  LegateCheck(BackendNetwork::coll_inited == true);
  for (MPI_Comm& mpi_comm : mpi_comms) {
    CHECK_MPI(MPI_Comm_free(&mpi_comm));
  }
  mpi_comms.clear();
  int fina_flag = 0;
  CHECK_MPI(MPI_Finalized(&fina_flag));
  if (fina_flag == 1) {
    LEGATE_ABORT("MPI should not have been finalized");
  }
  if (self_init_mpi) {
    CHECK_MPI(MPI_Finalize());
    detail::log_coll().info("finalize mpi");
  }
  BackendNetwork::coll_inited = false;
}

void MPINetwork::abort()
{
  int init = 0;

  static_cast<void>(MPI_Initialized(&init));
  if (init) {
    // noreturn
    static_cast<void>(MPI_Abort(MPI_COMM_WORLD, 1));
  }
}

int MPINetwork::init_comm()
{
  int id = 0;
  CHECK_COLL(collGetUniqueId(&id));
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    int send_id = id;
    // check if all ranks get the same unique id
    CHECK_MPI(MPI_Bcast(&send_id, 1, MPI_INT, 0, MPI_COMM_WORLD));
    LegateCheck(send_id == id);
  }
  LegateCheck(static_cast<int>(mpi_comms.size()) == id);
  // create mpi comm
  MPI_Comm mpi_comm;
  CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
  mpi_comms.push_back(mpi_comm);
  detail::log_coll().debug("Init comm id %d", id);
  return id;
}

int MPINetwork::comm_create(CollComm global_comm,
                            int global_comm_size,
                            int global_rank,
                            int unique_id,
                            const int* mapping_table)
{
  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank      = global_rank;
  global_comm->status           = true;
  global_comm->unique_id        = unique_id;
  int mpi_rank, mpi_comm_size;
  int compare_result;
  MPI_Comm comm = mpi_comms[unique_id];
  CHECK_MPI(MPI_Comm_compare(comm, MPI_COMM_WORLD, &compare_result));
  LegateCheck(MPI_CONGRUENT == compare_result);

  CHECK_MPI(MPI_Comm_rank(comm, &mpi_rank));
  CHECK_MPI(MPI_Comm_size(comm, &mpi_comm_size));
  global_comm->mpi_comm_size = mpi_comm_size;
  global_comm->mpi_rank      = mpi_rank;
  global_comm->mpi_comm      = comm;
  LegateCheck(mapping_table != nullptr);
  legate::detail::typed_malloc(&(global_comm->mapping_table.global_rank), global_comm_size);
  legate::detail::typed_malloc(&(global_comm->mapping_table.mpi_rank), global_comm_size);
  std::memcpy(global_comm->mapping_table.mpi_rank, mapping_table, sizeof(int) * global_comm_size);
  for (int i = 0; i < global_comm_size; i++) {
    global_comm->mapping_table.global_rank[i] = i;
  }
  std::tie(global_comm->nb_threads, global_comm->mpi_comm_size_actual) =
    mostFrequent(mapping_table, global_comm_size);
  return CollSuccess;
}

int MPINetwork::comm_destroy(CollComm global_comm)
{
  if (global_comm->mapping_table.global_rank != nullptr) {
    std::free(global_comm->mapping_table.global_rank);
    global_comm->mapping_table.global_rank = nullptr;
  }
  if (global_comm->mapping_table.mpi_rank != nullptr) {
    std::free(global_comm->mapping_table.mpi_rank);
    global_comm->mapping_table.mpi_rank = nullptr;
  }
  global_comm->status = false;
  return CollSuccess;
}

int MPINetwork::alltoallv(const void* sendbuf,
                          const int sendcounts[],
                          const int sdispls[],
                          void* recvbuf,
                          const int recvcounts[],
                          const int rdispls[],
                          CollDataType type,
                          CollComm global_comm)
{
  MPI_Status status;

  const int total_size  = global_comm->global_comm_size;
  const int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  MPI_Aint lb, type_extent;
  CHECK_MPI(MPI_Type_get_extent(mpi_type, &lb, &type_extent));

  int sendto_global_rank, recvfrom_global_rank, sendto_mpi_rank, recvfrom_mpi_rank;
  for (int i = 1; i < total_size + 1; i++) {
    sendto_global_rank   = (global_rank + i) % total_size;
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    char* src            = static_cast<char*>(const_cast<void*>(sendbuf)) +
                static_cast<std::ptrdiff_t>(sdispls[sendto_global_rank]) * type_extent;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<std::ptrdiff_t>(rdispls[recvfrom_global_rank]) * type_extent;
    const int scount  = sendcounts[sendto_global_rank];
    const int rcount  = recvcounts[recvfrom_global_rank];
    sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    LegateCheck(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    LegateCheck(recvfrom_global_rank ==
                global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    const int send_tag = generateAlltoallvTag(sendto_global_rank, global_rank, global_comm);
    const int recv_tag = generateAlltoallvTag(global_rank, recvfrom_global_rank, global_comm);
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "AlltoallvMPI i: %d === global_rank %d, mpi rank %d, send to %d (%d), send_tag %d, "
        "recv from %d (%d), "
        "recv_tag %d",
        i,
        global_rank,
        global_comm->mpi_rank,
        sendto_global_rank,
        sendto_mpi_rank,
        send_tag,
        recvfrom_global_rank,
        recvfrom_mpi_rank,
        recv_tag);
    }
    CHECK_MPI(MPI_Sendrecv(src,
                           scount,
                           mpi_type,
                           sendto_mpi_rank,
                           send_tag,
                           dst,
                           rcount,
                           mpi_type,
                           recvfrom_mpi_rank,
                           recv_tag,
                           global_comm->mpi_comm,
                           &status));
  }

  return CollSuccess;
}

int MPINetwork::alltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  const auto total_size  = global_comm->global_comm_size;
  const auto global_rank = global_comm->global_rank;
  const auto mpi_type    = dtypeToMPIDtype(type);

  MPI_Aint lb, type_extent;
  CHECK_MPI(MPI_Type_get_extent(mpi_type, &lb, &type_extent));

  for (int i = 1; i < total_size + 1; i++) {
    const auto sendto_global_rank   = (global_rank + i) % total_size;
    const auto recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    const char* src                 = static_cast<char*>(const_cast<void*>(sendbuf)) +
                      static_cast<std::ptrdiff_t>(sendto_global_rank) * type_extent * count;
    char* dst = static_cast<char*>(recvbuf) +
                static_cast<std::ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
    const auto sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    const auto recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];
    LegateCheck(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    LegateCheck(recvfrom_global_rank ==
                global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    const int send_tag = generateAlltoallTag(sendto_global_rank, global_rank, global_comm);
    const int recv_tag = generateAlltoallTag(global_rank, recvfrom_global_rank, global_comm);
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "AlltoallMPI i: %d === global_rank %d, mpi rank %d, send to %d (%d), send_tag %d, "
        "recv from %d (%d), "
        "recv_tag %d",
        i,
        global_rank,
        global_comm->mpi_rank,
        sendto_global_rank,
        sendto_mpi_rank,
        send_tag,
        recvfrom_global_rank,
        recvfrom_mpi_rank,
        recv_tag);
    }
    MPI_Status status;

    CHECK_MPI(MPI_Sendrecv(src,
                           count,
                           mpi_type,
                           sendto_mpi_rank,
                           send_tag,
                           dst,
                           count,
                           mpi_type,
                           recvfrom_mpi_rank,
                           recv_tag,
                           global_comm->mpi_comm,
                           &status));
  }

  return CollSuccess;
}

int MPINetwork::allgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  const int total_size = global_comm->global_comm_size;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  MPI_Aint lb, type_extent;
  CHECK_MPI(MPI_Type_get_extent(mpi_type, &lb, &type_extent));

  void* sendbuf_tmp = const_cast<void*>(sendbuf);

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = allocateInplaceBuffer(recvbuf, type_extent * count);
  }

  auto guard = legate::detail::make_scope_guard([&] {
    if (sendbuf == recvbuf) {
      std::free(sendbuf_tmp);
    }
  });

  CHECK_COLL(gather(sendbuf_tmp, recvbuf, count, type, 0, global_comm));
  CHECK_COLL(bcast(recvbuf, count * total_size, type, 0, global_comm));
  return CollSuccess;
}

int MPINetwork::gather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, int root, CollComm global_comm)
{
  MPI_Status status;

  const int total_size  = global_comm->global_comm_size;
  const int global_rank = global_comm->global_rank;

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  // Should not see inplace here
  if (sendbuf == recvbuf) {
    throw std::invalid_argument{"MPINetwork::gather() does not support inplace gather"};
  }

  const int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  LegateCheck(root == global_comm->mapping_table.global_rank[root]);

  int tag;

  // non-root
  if (global_rank != root) {
    tag = generateGatherTag(global_rank, global_comm);
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "GatherMPI: non-root send global_rank %d, mpi rank %d, send to %d (%d), tag %d",
        global_rank,
        global_comm->mpi_rank,
        root,
        root_mpi_rank,
        tag);
    }
    CHECK_MPI(MPI_Send(sendbuf, count, mpi_type, root_mpi_rank, tag, global_comm->mpi_comm));
    return CollSuccess;
  }

  // root
  MPI_Aint incr, lb, type_extent;
  CHECK_MPI(MPI_Type_get_extent(mpi_type, &lb, &type_extent));
  incr      = type_extent * static_cast<std::ptrdiff_t>(count);
  char* dst = static_cast<char*>(recvbuf);
  int recvfrom_mpi_rank;
  for (int i = 0; i < total_size; i++) {
    recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    LegateCheck(i == global_comm->mapping_table.global_rank[i]);
    tag = generateGatherTag(i, global_comm);
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "GatherMPI: root i %d === global_rank %d, mpi rank %d, recv %p, from %d (%d), tag %d",
        i,
        global_rank,
        global_comm->mpi_rank,
        static_cast<void*>(dst),
        i,
        recvfrom_mpi_rank,
        tag);
    }
    LegateCheck(dst != nullptr);
    if (global_rank == i) {
      std::memcpy(dst, sendbuf, incr);
    } else {
      CHECK_MPI(
        MPI_Recv(dst, count, mpi_type, recvfrom_mpi_rank, tag, global_comm->mpi_comm, &status));
    }
    dst += incr;
  }

  return CollSuccess;
}

int MPINetwork::bcast(void* buf, int count, CollDataType type, int root, CollComm global_comm)
{
  int tag;
  MPI_Status status;

  const int total_size  = global_comm->global_comm_size;
  const int global_rank = global_comm->global_rank;

  const int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  LegateCheck(root == global_comm->mapping_table.global_rank[root]);

  MPI_Datatype mpi_type = dtypeToMPIDtype(type);

  // non-root
  if (global_rank != root) {
    tag = generateBcastTag(global_rank, global_comm);
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "BcastMPI: non-root recv global_rank %d, mpi rank %d, send to %d (%d), tag %d",
        global_rank,
        global_comm->mpi_rank,
        root,
        root_mpi_rank,
        tag);
    }
    CHECK_MPI(MPI_Recv(buf, count, mpi_type, root_mpi_rank, tag, global_comm->mpi_comm, &status));
    return CollSuccess;
  }

  // root
  int sendto_mpi_rank;
  for (int i = 0; i < total_size; i++) {
    sendto_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    LegateCheck(i == global_comm->mapping_table.global_rank[i]);
    tag = generateBcastTag(i, global_comm);
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "BcastMPI: root i %d === global_rank %d, mpi rank %d, send to %d (%d), tag %d",
        i,
        global_rank,
        global_comm->mpi_rank,
        i,
        sendto_mpi_rank,
        tag);
    }
    if (global_rank != i) {
      CHECK_MPI(MPI_Send(buf, count, mpi_type, sendto_mpi_rank, tag, global_comm->mpi_comm));
    }
  }

  return CollSuccess;
}

// protected functions start from here

MPI_Datatype MPINetwork::dtypeToMPIDtype(CollDataType dtype)
{
  switch (dtype) {
    case CollDataType::CollInt8: {
      return MPI_INT8_T;
    }
    case CollDataType::CollChar: {
      return MPI_CHAR;
    }
    case CollDataType::CollUint8: {
      return MPI_UINT8_T;
    }
    case CollDataType::CollInt: {
      return MPI_INT;
    }
    case CollDataType::CollUint32: {
      return MPI_UINT32_T;
    }
    case CollDataType::CollInt64: {
      return MPI_INT64_T;
    }
    case CollDataType::CollUint64: {
      return MPI_UINT64_T;
    }
    case CollDataType::CollFloat: {
      return MPI_FLOAT;
    }
    case CollDataType::CollDouble: {
      return MPI_DOUBLE;
    }
    default: {
      LEGATE_ABORT("Unknown datatype");
      return MPI_BYTE;
    }
  }
}

int MPINetwork::generateAlltoallTag(int rank1, int rank2, CollComm global_comm) const
{
  const int tag = match2ranks(rank1, rank2, global_comm) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG;
  LegateCheck(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

int MPINetwork::generateAlltoallvTag(int rank1, int rank2, CollComm global_comm) const
{
  const int tag =
    match2ranks(rank1, rank2, global_comm) * CollTag::MAX_TAG + CollTag::ALLTOALLV_TAG;
  LegateCheck(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

int MPINetwork::generateBcastTag(int rank, CollComm /*global_comm*/) const
{
  const int tag = rank * CollTag::MAX_TAG + CollTag::BCAST_TAG;
  LegateCheck(tag <= mpi_tag_ub && tag >= 0);
  return tag;
}

int MPINetwork::generateGatherTag(int rank, CollComm /*global_comm*/) const
{
  const int tag = rank * CollTag::MAX_TAG + CollTag::GATHER_TAG;
  LegateCheck(tag <= mpi_tag_ub && tag > 0);
  return tag;
}

}  // namespace legate::comm::coll
