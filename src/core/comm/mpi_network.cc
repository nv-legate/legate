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

#include "core/comm/mpi_network.h"

#include "core/comm/coll.h"
#include "core/utilities/macros.h"
#include "core/utilities/scope_guard.h"
#include "core/utilities/span.h"
#include "core/utilities/typedefs.h"

#include "legate_defines.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <unordered_map>

namespace legate::comm::coll {

#define LEGATE_CHECK_MPI(...)                                                                     \
  do {                                                                                            \
    const int lgcore_check_mpi_result_ = __VA_ARGS__;                                             \
    if (LEGATE_UNLIKELY(lgcore_check_mpi_result_ != MPI_SUCCESS)) {                               \
      LEGATE_ABORT("Internal MPI failure with error code "                                        \
                   << lgcore_check_mpi_result_ << " in " << __FILE__ << ":" << __LINE__ << " in " \
                   << __func__ << "(): " << LEGATE_STRINGIZE(__VA_ARGS__));                       \
    }                                                                                             \
  } while (0)

// public functions start from here

MPINetwork::MPINetwork(int /*argc*/, char* /*argv*/[])
{
  detail::log_coll().debug() << "Enable MPINetwork";
  LEGATE_CHECK(current_unique_id_ == 0);

  int init_flag = 0;

  LEGATE_CHECK_MPI(MPI_Initialized(&init_flag));
  if (!init_flag) {
    int provided;

    detail::log_coll().info() << "MPI being initialized by legate";
    LEGATE_CHECK_MPI(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
    self_init_mpi_ = true;
  }

  int mpi_thread_model;

  LEGATE_CHECK_MPI(MPI_Query_thread(&mpi_thread_model));
  if (mpi_thread_model != MPI_THREAD_MULTIPLE) {
    LEGATE_ABORT(
      "MPI has been initialized by others, but is not initialized with "
      "MPI_THREAD_MULTIPLE");
  }

  int flag    = 0;
  int* tag_ub = nullptr;
  // check
  LEGATE_CHECK_MPI(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_ub, &flag));
  LEGATE_CHECK(flag);
  mpi_tag_ub_ = *tag_ub;
  LEGATE_CHECK(mpi_comms_.empty());
  BackendNetwork::coll_inited_ = true;
  BackendNetwork::comm_type    = CollCommType::CollMPI;
}

MPINetwork::~MPINetwork()
{
  detail::log_coll().debug("Finalize MPINetwork");
  LEGATE_CHECK(BackendNetwork::coll_inited_ == true);

  int finalized = 0;
  LEGATE_CHECK_MPI(MPI_Finalized(&finalized));
  if (finalized == 1) {
    LEGATE_ABORT("MPI should not have been finalized");
  }

  for (MPI_Comm& mpi_comm : mpi_comms_) {
    LEGATE_CHECK_MPI(MPI_Comm_free(&mpi_comm));
  }
  mpi_comms_.clear();

  if (self_init_mpi_) {
    detail::log_coll().info() << "finalize mpi";
    LEGATE_CHECK_MPI(MPI_Finalize());
  }
  BackendNetwork::coll_inited_ = false;
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
  auto id = get_unique_id_();
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    int send_id = id;
    // check if all ranks get the same unique id
    LEGATE_CHECK_MPI(MPI_Bcast(&send_id, 1, MPI_INT, 0, MPI_COMM_WORLD));
    LEGATE_CHECK(send_id == id);
  }
  LEGATE_CHECK(static_cast<int>(mpi_comms_.size()) == id);
  // create mpi comm
  MPI_Comm mpi_comm;
  LEGATE_CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
  mpi_comms_.push_back(mpi_comm);
  detail::log_coll().debug() << "Init comm id " << id;
  return id;
}

namespace {

[[nodiscard]] std::pair<int, int> most_frequent(Span<const int> arr)
{
  std::unordered_map<int, int> hash;

  for (auto&& v : arr) {
    ++hash[v];
  }

  const auto max_elem = std::max_element(hash.begin(), hash.end(), [](auto&& p1, auto&& p2) {
                          return p1.second < p2.second;
                        })->second;

  return {max_elem, hash.size()};
}

}  // namespace

void MPINetwork::comm_create(CollComm global_comm,
                             int global_comm_size,
                             int global_rank,
                             int unique_id,
                             const int* mapping_table)
{
  LEGATE_ASSERT(global_comm_size >= 0);
  LEGATE_CHECK(mapping_table != nullptr);

  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank      = global_rank;
  global_comm->status           = true;
  global_comm->unique_id        = unique_id;
  global_comm->mpi_comm         = mpi_comms_[unique_id];

  int compare_result;

  LEGATE_CHECK_MPI(MPI_Comm_compare(global_comm->mpi_comm, MPI_COMM_WORLD, &compare_result));
  LEGATE_CHECK(MPI_CONGRUENT == compare_result);
  LEGATE_CHECK_MPI(MPI_Comm_rank(global_comm->mpi_comm, &global_comm->mpi_rank));
  LEGATE_CHECK_MPI(MPI_Comm_size(global_comm->mpi_comm, &global_comm->mpi_comm_size));

  auto global_ranks = std::make_unique<int[]>(global_comm_size);
  auto mpi_ranks    = std::make_unique<int[]>(global_comm_size);
  std::memcpy(mpi_ranks.get(),
              mapping_table,
              static_cast<std::size_t>(global_comm_size * sizeof(*mapping_table)));
  std::iota(global_ranks.get(), global_ranks.get() + global_comm_size, 0);
  std::tie(global_comm->nb_threads, global_comm->mpi_comm_size_actual) =
    most_frequent({mapping_table, static_cast<std::size_t>(global_comm_size)});

  global_comm->mapping_table.global_rank = global_ranks.release();
  global_comm->mapping_table.mpi_rank    = mpi_ranks.release();
}

void MPINetwork::comm_destroy(CollComm global_comm)
{
  delete[] std::exchange(global_comm->mapping_table.global_rank, nullptr);
  delete[] std::exchange(global_comm->mapping_table.mpi_rank, nullptr);
  global_comm->status = false;
}

void MPINetwork::all_to_all_v(const void* sendbuf,
                              const int sendcounts[],
                              const int sdispls[],
                              void* recvbuf,
                              const int recvcounts[],
                              const int rdispls[],
                              CollDataType type,
                              CollComm global_comm)
{
  const int total_size  = global_comm->global_comm_size;
  const int global_rank = global_comm->global_rank;
  const auto mpi_type   = dtype_to_mpi_dtype_(type);
  MPI_Aint lb, type_extent;

  LEGATE_CHECK_MPI(MPI_Type_get_extent(mpi_type, &lb, &type_extent));
  for (int i = 1; i < total_size + 1; i++) {
    const auto sendto_global_rank   = (global_rank + i) % total_size;
    const auto recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // How else are we supposed to make a char* out of const void*?????????????????????????
    // NOLINTNEXTLINE(bugprone-casting-through-void)
    const auto src = static_cast<char*>(const_cast<void*>(sendbuf)) +
                     static_cast<std::ptrdiff_t>(sdispls[sendto_global_rank]) * type_extent;
    const auto dst = static_cast<char*>(recvbuf) +
                     static_cast<std::ptrdiff_t>(rdispls[recvfrom_global_rank]) * type_extent;
    const int scount             = sendcounts[sendto_global_rank];
    const int rcount             = recvcounts[recvfrom_global_rank];
    const auto sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    const auto recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];

    LEGATE_CHECK(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    LEGATE_CHECK(recvfrom_global_rank ==
                 global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    const int send_tag = generate_alltoallv_tag_(sendto_global_rank, global_rank, global_comm);
    const int recv_tag = generate_alltoallv_tag_(global_rank, recvfrom_global_rank, global_comm);

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug() << "AlltoallvMPI i: " << i << " === global_rank " << global_rank
                                 << ", mpi rank " << global_comm->mpi_rank << ", send to "
                                 << sendto_global_rank << " (" << sendto_mpi_rank << "), send_tag "
                                 << send_tag
                                 << ", "
                                    "recv from "
                                 << recvfrom_global_rank << " (" << recvfrom_mpi_rank
                                 << "), "
                                    "recv_tag "
                                 << recv_tag;
    }

    MPI_Status status;

    LEGATE_CHECK_MPI(MPI_Sendrecv(src,
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
}

void MPINetwork::all_to_all(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  const auto total_size  = global_comm->global_comm_size;
  const auto global_rank = global_comm->global_rank;
  const auto mpi_type    = dtype_to_mpi_dtype_(type);
  MPI_Aint lb, type_extent;

  LEGATE_CHECK_MPI(MPI_Type_get_extent(mpi_type, &lb, &type_extent));
  for (int i = 1; i < total_size + 1; i++) {
    const auto sendto_global_rank   = (global_rank + i) % total_size;
    const auto recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // How else are we supposed to make a char* out of const void*?????????????????????????
    // NOLINTNEXTLINE(bugprone-casting-through-void)
    const auto src = static_cast<char*>(const_cast<void*>(sendbuf)) +
                     static_cast<std::ptrdiff_t>(sendto_global_rank) * type_extent * count;
    const auto dst = static_cast<char*>(recvbuf) +
                     static_cast<std::ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
    const auto sendto_mpi_rank   = global_comm->mapping_table.mpi_rank[sendto_global_rank];
    const auto recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[recvfrom_global_rank];

    LEGATE_CHECK(sendto_global_rank == global_comm->mapping_table.global_rank[sendto_global_rank]);
    LEGATE_CHECK(recvfrom_global_rank ==
                 global_comm->mapping_table.global_rank[recvfrom_global_rank]);
    // tag: seg idx + rank_idx + tag
    const int send_tag = generate_alltoall_tag_(sendto_global_rank, global_rank, global_comm);
    const int recv_tag = generate_alltoall_tag_(global_rank, recvfrom_global_rank, global_comm);

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug() << "AlltoallMPI i: " << i << " === global_rank " << global_rank
                                 << ", mpi rank " << global_comm->mpi_rank << ", send to "
                                 << sendto_global_rank << " (" << sendto_mpi_rank << "), send_tag "
                                 << send_tag << ", recv from " << recvfrom_global_rank << " ("
                                 << recvfrom_mpi_rank << "), recv_tag " << recv_tag;
    }

    MPI_Status status;

    LEGATE_CHECK_MPI(MPI_Sendrecv(src,
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
}

void MPINetwork::all_gather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  const int total_size = global_comm->global_comm_size;
  const auto mpi_type  = dtype_to_mpi_dtype_(type);
  auto sendbuf_tmp     = const_cast<void*>(sendbuf);
  MPI_Aint lb, type_extent;

  LEGATE_CHECK_MPI(MPI_Type_get_extent(mpi_type, &lb, &type_extent));

  const auto num_bytes = static_cast<std::size_t>(type_extent * count);
  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = allocate_inplace_buffer_(recvbuf, num_bytes);
  }
  // For some reason clang-format like to pack this one along one line...
  // clang-format off
  LEGATE_SCOPE_GUARD(
    if (sendbuf == recvbuf) {
      delete_inplace_buffer_(sendbuf_tmp, num_bytes);
    }
  );
  // clang-format on

  gather_(sendbuf_tmp, recvbuf, count, type, 0, global_comm);
  bcast_(recvbuf, count * total_size, type, 0, global_comm);
}

void MPINetwork::gather_(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, int root, CollComm global_comm)
{
  MPI_Status status;
  const int total_size  = global_comm->global_comm_size;
  const int global_rank = global_comm->global_rank;
  const auto mpi_type   = dtype_to_mpi_dtype_(type);

  // Should not see inplace here
  if (sendbuf == recvbuf) {
    throw std::invalid_argument{"MPINetwork::gather() does not support inplace gather"};
  }

  const int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  LEGATE_CHECK(root == global_comm->mapping_table.global_rank[root]);

  // non-root
  if (global_rank != root) {
    const auto tag = generate_gather_tag_(global_rank, global_comm);

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug() << "GatherMPI: non-root send global_rank " << global_rank
                                 << ", mpi rank " << global_comm->mpi_rank << ", send to " << root
                                 << " (" << root_mpi_rank << "), tag " << tag;
    }
    LEGATE_CHECK_MPI(MPI_Send(sendbuf, count, mpi_type, root_mpi_rank, tag, global_comm->mpi_comm));
  }

  // root
  MPI_Aint lb, type_extent;

  LEGATE_CHECK_MPI(MPI_Type_get_extent(mpi_type, &lb, &type_extent));

  const auto incr = static_cast<std::size_t>(type_extent * static_cast<std::ptrdiff_t>(count));
  char* dst       = static_cast<char*>(recvbuf);
  LEGATE_CHECK(dst != nullptr);
  for (int i = 0; i < total_size; i++) {
    const auto recvfrom_mpi_rank = global_comm->mapping_table.mpi_rank[i];
    LEGATE_CHECK(i == global_comm->mapping_table.global_rank[i]);
    const auto tag = generate_gather_tag_(i, global_comm);

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug() << "GatherMPI: root i " << i << " === global_rank " << global_rank
                                 << ", mpi rank " << global_comm->mpi_rank << ", recv " << dst
                                 << ", from " << i << " (" << recvfrom_mpi_rank << "), tag " << tag;
    }
    if (global_rank == i) {
      std::memcpy(dst, sendbuf, incr);
    } else {
      LEGATE_CHECK_MPI(
        MPI_Recv(dst, count, mpi_type, recvfrom_mpi_rank, tag, global_comm->mpi_comm, &status));
    }
    dst += incr;
  }
}

void MPINetwork::bcast_(void* buf, int count, CollDataType type, int root, CollComm global_comm)
{
  const int total_size    = global_comm->global_comm_size;
  const int global_rank   = global_comm->global_rank;
  const int root_mpi_rank = global_comm->mapping_table.mpi_rank[root];
  const auto mpi_type     = dtype_to_mpi_dtype_(type);

  LEGATE_CHECK(root == global_comm->mapping_table.global_rank[root]);
  // non-root
  if (global_rank != root) {
    const auto tag = generate_bcast_tag_(global_rank, global_comm);

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "BcastMPI: non-root recv global_rank %d, mpi rank %d, send to %d (%d), tag %d",
        global_rank,
        global_comm->mpi_rank,
        root,
        root_mpi_rank,
        tag);
    }
    MPI_Status status;

    LEGATE_CHECK_MPI(
      MPI_Recv(buf, count, mpi_type, root_mpi_rank, tag, global_comm->mpi_comm, &status));
  }

  // root
  for (int i = 0; i < total_size; i++) {
    const auto sendto_mpi_rank = global_comm->mapping_table.mpi_rank[i];

    LEGATE_CHECK(i == global_comm->mapping_table.global_rank[i]);
    const auto tag = generate_bcast_tag_(i, global_comm);
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug() << "BcastMPI: root i " << i << " === global_rank " << global_rank
                                 << ", mpi rank " << global_comm->mpi_rank << ", send to " << i
                                 << " (" << sendto_mpi_rank << "), tag " << tag;
    }
    if (global_rank != i) {
      LEGATE_CHECK_MPI(MPI_Send(buf, count, mpi_type, sendto_mpi_rank, tag, global_comm->mpi_comm));
    }
  }
}

// protected functions start from here

MPI_Datatype MPINetwork::dtype_to_mpi_dtype_(CollDataType dtype)
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

namespace {

enum CollTag : std::uint8_t {
  BCAST_TAG     = 0,
  GATHER_TAG    = 1,
  ALLTOALL_TAG  = 2,
  ALLTOALLV_TAG = 3,
  MAX_TAG       = 10,
};

[[nodiscard]] int match_to_ranks(int rank1, int rank2, CollComm global_comm)
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

}  // namespace

int MPINetwork::generate_alltoall_tag_(int rank1, int rank2, CollComm global_comm) const
{
  const int tag =
    match_to_ranks(rank1, rank2, global_comm) * CollTag::MAX_TAG + CollTag::ALLTOALL_TAG;
  LEGATE_CHECK(tag <= mpi_tag_ub_ && tag > 0);
  return tag;
}  // namespace

int MPINetwork::generate_alltoallv_tag_(int rank1, int rank2, CollComm global_comm) const
{
  const int tag =
    match_to_ranks(rank1, rank2, global_comm) * CollTag::MAX_TAG + CollTag::ALLTOALLV_TAG;
  LEGATE_CHECK(tag <= mpi_tag_ub_ && tag > 0);
  return tag;
}

int MPINetwork::generate_bcast_tag_(int rank, CollComm /*global_comm*/) const
{
  const int tag = rank * CollTag::MAX_TAG + CollTag::BCAST_TAG;
  LEGATE_CHECK(tag <= mpi_tag_ub_ && tag >= 0);
  return tag;
}

int MPINetwork::generate_gather_tag_(int rank, CollComm /*global_comm*/) const
{
  const int tag = rank * CollTag::MAX_TAG + CollTag::GATHER_TAG;
  LEGATE_CHECK(tag <= mpi_tag_ub_ && tag > 0);
  return tag;
}

}  // namespace legate::comm::coll
