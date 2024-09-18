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

#include "legate/comm/detail/local_network.h"

#include "legate_defines.h"

#include "legate/comm/coll.h"
#include "legate/comm/detail/logger.h"
#include "legate/utilities/assert.h"
#include "legate/utilities/macros.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace legate::detail::comm::coll {

// public functions start from here

LocalNetwork::LocalNetwork(int /*argc*/, char* /*argv*/[])
{
  logger().debug() << "Enable LocalNetwork";
  LEGATE_CHECK(current_unique_id_ == 0);
  LEGATE_CHECK(thread_comms_.empty());
  BackendNetwork::coll_inited_ = true;
  BackendNetwork::comm_type    = legate::comm::coll::CollCommType::CollLocal;
}

LocalNetwork::~LocalNetwork()
{
  logger().debug() << "Finalize LocalNetwork";
  LEGATE_CHECK(BackendNetwork::coll_inited_ == true);
  for (auto&& thread_comm : thread_comms_) {
    LEGATE_CHECK(!thread_comm->ready());
  }
  BackendNetwork::coll_inited_ = false;
}

int LocalNetwork::init_comm()
{
  auto id = get_unique_id_();
  LEGATE_CHECK(id >= 0 && thread_comms_.size() == static_cast<std::size_t>(id));
  // create thread comm
  thread_comms_.emplace_back(std::make_unique<ThreadComm>());
  logger().debug() << "Init comm id " << id;
  return id;
}

void LocalNetwork::comm_create(legate::comm::coll::CollComm global_comm,
                               int global_comm_size,
                               int global_rank,
                               int unique_id,
                               const int*)
{
  global_comm->global_comm_size     = global_comm_size;
  global_comm->global_rank          = global_rank;
  global_comm->status               = true;
  global_comm->unique_id            = unique_id;
  global_comm->mpi_comm_size        = 1;
  global_comm->mpi_comm_size_actual = 1;
  global_comm->mpi_rank             = 0;
  if (global_comm->global_rank == 0) {
    thread_comms_[global_comm->unique_id]->init(global_comm->global_comm_size);
  }
  while (!thread_comms_[global_comm->unique_id]->ready()) {}
  global_comm->local_comm = thread_comms_[global_comm->unique_id].get();
  barrier_local_(global_comm);
  LEGATE_CHECK(global_comm->local_comm->ready());
  global_comm->nb_threads = global_comm->global_comm_size;
}

void LocalNetwork::comm_destroy(legate::comm::coll::CollComm global_comm)
{
  const auto id = global_comm->unique_id;

  LEGATE_ASSERT(id >= 0);
  barrier_local_(global_comm);
  thread_comms_[static_cast<std::size_t>(id)]->finalize(global_comm->global_comm_size,
                                                        global_comm->global_rank == 0);
  global_comm->status = false;
}

void LocalNetwork::all_to_all_v(const void* sendbuf,
                                const int /*sendcounts*/[],
                                const int sdispls[],
                                void* recvbuf,
                                const int recvcounts[],
                                const int rdispls[],
                                legate::comm::coll::CollDataType type,
                                legate::comm::coll::CollComm global_comm)
{
  const auto total_size      = global_comm->global_comm_size;
  const auto global_rank     = global_comm->global_rank;
  const auto type_extent     = get_dtype_size_(type);
  const auto recvfrom_seg_id = global_rank;
  auto* loc_buffers          = global_comm->local_comm->buffers();
  auto* loc_displs           = global_comm->local_comm->displs();

  loc_displs[global_rank]  = sdispls;
  loc_buffers[global_rank] = sendbuf;
  for (int i = 1; i < total_size + 1; i++) {
    const auto recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // wait for other threads to update the buffer address
    while (loc_buffers[recvfrom_global_rank] == nullptr ||
           loc_displs[recvfrom_global_rank] == nullptr) {}

    // NOLINTBEGIN(bugprone-casting-through-void)
    const auto* src_base =
      static_cast<const char*>(static_cast<const void*>(loc_buffers[recvfrom_global_rank]));
    // NOLINTEND(bugprone-casting-through-void)
    const int* displs = loc_displs[recvfrom_global_rank];
    const auto* src   = static_cast<const void*>(
      src_base + static_cast<std::ptrdiff_t>(displs[recvfrom_seg_id]) * type_extent);
    auto* dst = static_cast<char*>(recvbuf) +
                static_cast<std::ptrdiff_t>(rdispls[recvfrom_global_rank]) * type_extent;
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      logger().debug() << "AlltoallvLocal i: " << i << " === global_rank " << global_rank
                       << ", dtype " << type_extent << ", copy rank " << recvfrom_global_rank
                       << " (seg " << recvfrom_seg_id << ", sdispls " << sdispls[recvfrom_seg_id]
                       << ", " << src << ") to rank " << global_rank << " (seg "
                       << recvfrom_global_rank << ", rdispls " << rdispls[recvfrom_global_rank]
                       << ", " << dst << ')';
    }
    std::memcpy(dst, src, recvcounts[recvfrom_global_rank] * type_extent);
  }

  barrier_local_(global_comm);
  __sync_synchronize();
  reset_local_buffer_(global_comm);
  barrier_local_(global_comm);
}

void LocalNetwork::all_to_all(const void* sendbuf,
                              void* recvbuf,
                              int count,
                              legate::comm::coll::CollDataType type,
                              legate::comm::coll::CollComm global_comm)
{
  LEGATE_CHECK(count >= 0);
  const auto total_size      = global_comm->global_comm_size;
  const auto global_rank     = global_comm->global_rank;
  const auto type_extent     = get_dtype_size_(type);
  const auto num_bytes       = type_extent * static_cast<std::size_t>(count);
  const auto recvfrom_seg_id = global_rank;
  const void* src_base       = nullptr;
  auto* buffers              = global_comm->local_comm->buffers();

  buffers[global_rank] = sendbuf;
  for (int i = 1; i < total_size + 1; i++) {
    const auto recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // wait for other threads to update the buffer address
    while (buffers[recvfrom_global_rank] == nullptr) {}
    src_base = buffers[recvfrom_global_rank];
    const auto* src =
      static_cast<const void*>(static_cast<const char*>(src_base) +
                               static_cast<std::ptrdiff_t>(recvfrom_seg_id) * num_bytes);
    auto* dst =
      static_cast<char*>(recvbuf) + static_cast<std::ptrdiff_t>(recvfrom_global_rank) * num_bytes;
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      logger().debug() << "AlltoallLocal i: " << i << " === global_rank " << global_rank
                       << ", dtype " << type_extent << ", copy rank " << recvfrom_global_rank
                       << " (seg " << recvfrom_seg_id << ", " << src << ") to rank " << global_rank
                       << "(seg " << recvfrom_global_rank << ", " << dst << ')';
    }
    std::memcpy(dst, src, num_bytes);
  }
  barrier_local_(global_comm);
  __sync_synchronize();
  reset_local_buffer_(global_comm);
  barrier_local_(global_comm);
}

void LocalNetwork::all_gather(const void* sendbuf,
                              void* recvbuf,
                              int count,
                              legate::comm::coll::CollDataType type,
                              legate::comm::coll::CollComm global_comm)
{
  LEGATE_CHECK(count >= 0);
  const auto total_size   = global_comm->global_comm_size;
  const auto global_rank  = global_comm->global_rank;
  const auto type_extent  = get_dtype_size_(type);
  const auto num_bytes    = type_extent * static_cast<std::size_t>(count);
  const auto* sendbuf_tmp = sendbuf;
  auto* buffers           = global_comm->local_comm->buffers();

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = allocate_inplace_buffer_(recvbuf, num_bytes);
  }

  buffers[global_rank] = sendbuf_tmp;
  for (int recvfrom_global_rank = 0; recvfrom_global_rank < total_size; recvfrom_global_rank++) {
    // wait for other threads to update the buffer address
    while (buffers[recvfrom_global_rank] == nullptr) {}
    const void* src = buffers[recvfrom_global_rank];
    char* dst =
      static_cast<char*>(recvbuf) + static_cast<std::ptrdiff_t>(recvfrom_global_rank) * num_bytes;
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      logger().debug() << "AllgatherLocal i: " << recvfrom_global_rank << " === global_rank "
                       << global_rank << ", dtype " << type_extent << ", copy rank "
                       << recvfrom_global_rank << " (" << src << ") to rank " << global_rank << " ("
                       << dst << ')';
    }
    std::memcpy(dst, src, num_bytes);
  }

  barrier_local_(global_comm);
  if (sendbuf == recvbuf) {
    delete_inplace_buffer_(const_cast<void*>(sendbuf_tmp), num_bytes);
  }
  __sync_synchronize();
  reset_local_buffer_(global_comm);
  barrier_local_(global_comm);
}

// protected functions start from here

std::size_t LocalNetwork::get_dtype_size_(legate::comm::coll::CollDataType dtype)
{
  switch (dtype) {
    case legate::comm::coll::CollDataType::CollInt8:
    case legate::comm::coll::CollDataType::CollChar: {
      return sizeof(char);
    }
    case legate::comm::coll::CollDataType::CollUint8: {
      return sizeof(std::uint8_t);
    }
    case legate::comm::coll::CollDataType::CollInt: {
      return sizeof(int);
    }
    case legate::comm::coll::CollDataType::CollUint32: {
      return sizeof(std::uint32_t);
    }
    case legate::comm::coll::CollDataType::CollInt64: {
      return sizeof(std::int64_t);
    }
    case legate::comm::coll::CollDataType::CollUint64: {
      return sizeof(std::uint64_t);
    }
    case legate::comm::coll::CollDataType::CollFloat: {
      return sizeof(float);
    }
    case legate::comm::coll::CollDataType::CollDouble: {
      return sizeof(double);
    }
    default: {
      LEGATE_ABORT("Unknown datatype");
      return 0;
    }
  }
}

void LocalNetwork::reset_local_buffer_(legate::comm::coll::CollComm global_comm)
{
  const auto global_rank                          = global_comm->global_rank;
  global_comm->local_comm->buffers()[global_rank] = nullptr;
  global_comm->local_comm->displs()[global_rank]  = nullptr;
}

void LocalNetwork::barrier_local_(legate::comm::coll::CollComm global_comm)
{
  LEGATE_CHECK(BackendNetwork::coll_inited_ == true);
  global_comm->local_comm->barrier_local();
}

}  // namespace legate::detail::comm::coll
