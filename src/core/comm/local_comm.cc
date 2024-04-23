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

#include "core/comm/coll.h"
#include "core/utilities/detail/malloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace legate::comm::coll {

void ThreadComm::init(std::int32_t global_comm_size)
{
  const auto uint_comm_size = static_cast<unsigned int>(global_comm_size);

  CHECK_PTHREAD_CALL_V(pthread_barrier_init(&barrier_, nullptr, uint_comm_size));
  // Allocate with unique_ptr so RAII cleans these up in case of exception
  std::unique_ptr<std::atomic<const void*>[]> buff_tmp{
    new std::atomic<const void*>[global_comm_size]};
  std::unique_ptr<const int*[]> displ_tmp{new const int*[global_comm_size]};

  buffers = buff_tmp.release();
  displs  = displ_tmp.release();
  for (std::int32_t i = 0; i < global_comm_size; ++i) {
    buffers[i] = nullptr;
    displs[i]  = nullptr;
  }
  entered_finalize_ = 0;
  ready_flag_       = true;
}

void ThreadComm::finalize(std::int32_t global_comm_size, bool is_finalizer)
{
  ++entered_finalize_;
  if (is_finalizer) {
    // Need to ensure that all other threads have left the barrier before we can destroy the
    // thread_comm.
    while (entered_finalize_ != global_comm_size) {}
    entered_finalize_ = 0;
    clear();
  } else {
    // The remaining threads are not allowed to leave until the finalizer thread has finished
    // its work.
    while (ready()) {}
  }
}

void ThreadComm::clear() noexcept
{
  CHECK_PTHREAD_CALL_V(pthread_barrier_destroy(&barrier_));
  delete[] std::exchange(buffers, nullptr);
  delete[] std::exchange(displs, nullptr);
  ready_flag_ = false;
}

void ThreadComm::barrier_local()
{
  if (const auto ret = pthread_barrier_wait(&barrier_)) {
    if (ret == PTHREAD_BARRIER_SERIAL_THREAD) {
      return;
    }
    CHECK_PTHREAD_CALL_V(ret);
  }
}

bool ThreadComm::ready() const { return ready_flag_; }

ThreadComm::~ThreadComm() noexcept
{
  delete[] buffers;
  delete[] displs;
}

// public functions start from here

LocalNetwork::LocalNetwork(int /*argc*/, char* /*argv*/[])
{
  detail::log_coll().debug("Enable LocalNetwork");
  LegateCheck(current_unique_id == 0);
  LegateCheck(thread_comms.empty());
  BackendNetwork::coll_inited = true;
  BackendNetwork::comm_type   = CollCommType::CollLocal;
}

LocalNetwork::~LocalNetwork()
{
  detail::log_coll().debug("Finalize LocalNetwork");
  LegateCheck(BackendNetwork::coll_inited == true);
  for (auto&& thread_comm : thread_comms) {
    LegateCheck(!thread_comm->ready());
  }
  BackendNetwork::coll_inited = false;
}

int LocalNetwork::comm_create(CollComm global_comm,
                              int global_comm_size,
                              int global_rank,
                              int unique_id,
                              const int* mapping_table)
{
  global_comm->global_comm_size = global_comm_size;
  global_comm->global_rank      = global_rank;
  global_comm->status           = true;
  global_comm->unique_id        = unique_id;
  LegateCheck(mapping_table == nullptr);
  global_comm->mpi_comm_size        = 1;
  global_comm->mpi_comm_size_actual = 1;
  global_comm->mpi_rank             = 0;
  if (global_comm->global_rank == 0) {
    thread_comms[global_comm->unique_id]->init(global_comm->global_comm_size);
  }
  while (!thread_comms[global_comm->unique_id]->ready()) {}
  global_comm->local_comm = thread_comms[global_comm->unique_id].get();
  barrierLocal(global_comm);
  LegateCheck(const_cast<const ThreadComm*>(global_comm->local_comm)->ready());
  global_comm->nb_threads = global_comm->global_comm_size;
  return CollSuccess;
}

int LocalNetwork::comm_destroy(CollComm global_comm)
{
  const auto id = global_comm->unique_id;

  LegateAssert(id >= 0);
  barrierLocal(global_comm);
  thread_comms[static_cast<std::size_t>(id)]->finalize(global_comm->global_comm_size,
                                                       global_comm->global_rank == 0);
  global_comm->status = false;
  return CollSuccess;
}

int LocalNetwork::init_comm()
{
  int id   = 0;
  auto ret = collGetUniqueId(&id);
  LegateCheck(ret == CollSuccess);
  LegateCheck(id >= 0 && thread_comms.size() == static_cast<std::size_t>(id));
  // create thread comm
  thread_comms.emplace_back(std::make_unique<ThreadComm>());
  detail::log_coll().debug("Init comm id %d", id);
  return id;
}

int LocalNetwork::alltoallv(const void* sendbuf,
                            const int /*sendcounts*/[],
                            const int sdispls[],
                            void* recvbuf,
                            const int recvcounts[],
                            const int rdispls[],
                            CollDataType type,
                            CollComm global_comm)
{
  const int total_size  = global_comm->global_comm_size;
  const int global_rank = global_comm->global_rank;

  const auto type_extent = getDtypeSize(type);

  global_comm->local_comm->displs[global_rank]  = sdispls;
  global_comm->local_comm->buffers[global_rank] = sendbuf;
  __sync_synchronize();

  int recvfrom_global_rank;
  const int recvfrom_seg_id = global_rank;
  const void* src_base      = nullptr;
  const int* displs         = nullptr;
  for (int i = 1; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // wait for other threads to update the buffer address
    while (global_comm->local_comm->buffers[recvfrom_global_rank] == nullptr ||
           static_cast<const int* volatile*>(
             global_comm->local_comm->displs)[recvfrom_global_rank] == nullptr) {}
    src_base = global_comm->local_comm->buffers[recvfrom_global_rank];
    displs   = global_comm->local_comm->displs[recvfrom_global_rank];
    const auto* src =
      static_cast<const void*>(static_cast<const char*>(src_base) +
                               static_cast<std::ptrdiff_t>(displs[recvfrom_seg_id]) * type_extent);
    auto* dst = static_cast<char*>(recvbuf) +
                static_cast<std::ptrdiff_t>(rdispls[recvfrom_global_rank]) * type_extent;
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "AlltoallvLocal i: %d === global_rank %d, dtype %zu, copy rank %d (seg %d, sdispls %d, %p) "
        "to rank %d (seg %d, rdispls %d, %p)",
        i,
        global_rank,
        type_extent,
        recvfrom_global_rank,
        recvfrom_seg_id,
        sdispls[recvfrom_seg_id],
        src,
        global_rank,
        recvfrom_global_rank,
        rdispls[recvfrom_global_rank],
        static_cast<void*>(dst));
    }
    std::memcpy(dst, src, recvcounts[recvfrom_global_rank] * type_extent);
  }

  barrierLocal(global_comm);

  __sync_synchronize();

  resetLocalBuffer(global_comm);
  barrierLocal(global_comm);

  return CollSuccess;
}

int LocalNetwork::alltoall(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  const int total_size  = global_comm->global_comm_size;
  const int global_rank = global_comm->global_rank;

  const auto type_extent = getDtypeSize(type);

  global_comm->local_comm->buffers[global_rank] = sendbuf;
  __sync_synchronize();

  int recvfrom_global_rank;
  const int recvfrom_seg_id = global_rank;
  const void* src_base      = nullptr;
  for (int i = 1; i < total_size + 1; i++) {
    recvfrom_global_rank = (global_rank + total_size - i) % total_size;
    // wait for other threads to update the buffer address
    while (global_comm->local_comm->buffers[recvfrom_global_rank] == nullptr) {}
    src_base = global_comm->local_comm->buffers[recvfrom_global_rank];
    const auto* src =
      static_cast<const void*>(static_cast<const char*>(src_base) +
                               static_cast<std::ptrdiff_t>(recvfrom_seg_id) * type_extent * count);
    auto* dst = static_cast<char*>(recvbuf) +
                static_cast<std::ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "AlltoallLocal i: %d === global_rank %d, dtype %zu, copy rank %d (seg %d, %p) to rank %d "
        "(seg %d, %p)",
        i,
        global_rank,
        type_extent,
        recvfrom_global_rank,
        recvfrom_seg_id,
        src,
        global_rank,
        recvfrom_global_rank,
        static_cast<void*>(dst));
    }
    std::memcpy(dst, src, count * type_extent);
  }

  barrierLocal(global_comm);

  __sync_synchronize();

  resetLocalBuffer(global_comm);
  barrierLocal(global_comm);

  return CollSuccess;
}

int LocalNetwork::allgather(
  const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm)
{
  const int total_size   = global_comm->global_comm_size;
  const int global_rank  = global_comm->global_rank;
  const auto type_extent = getDtypeSize(type);

  const void* sendbuf_tmp = sendbuf;

  // MPI_IN_PLACE
  if (sendbuf == recvbuf) {
    sendbuf_tmp = allocateInplaceBuffer(recvbuf, type_extent * count);
  }

  global_comm->local_comm->buffers[global_rank] = sendbuf_tmp;
  __sync_synchronize();

  for (int recvfrom_global_rank = 0; recvfrom_global_rank < total_size; recvfrom_global_rank++) {
    // wait for other threads to update the buffer address
    while (global_comm->local_comm->buffers[recvfrom_global_rank] == nullptr) {}
    const void* src = global_comm->local_comm->buffers[recvfrom_global_rank];
    char* dst       = static_cast<char*>(recvbuf) +
                static_cast<std::ptrdiff_t>(recvfrom_global_rank) * type_extent * count;
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      detail::log_coll().debug(
        "AllgatherLocal i: %d === global_rank %d, dtype %zu, copy rank %d (%p) to rank %d (%p)",
        recvfrom_global_rank,
        global_rank,
        type_extent,
        recvfrom_global_rank,
        src,
        global_rank,
        static_cast<void*>(dst));
    }
    std::memcpy(dst, src, count * type_extent);
  }

  barrierLocal(global_comm);
  if (sendbuf == recvbuf) {
    std::free(const_cast<void*>(sendbuf_tmp));
  }

  __sync_synchronize();

  resetLocalBuffer(global_comm);
  barrierLocal(global_comm);

  return CollSuccess;
}

// protected functions start from here

std::size_t LocalNetwork::getDtypeSize(CollDataType dtype)
{
  switch (dtype) {
    case CollDataType::CollInt8:
    case CollDataType::CollChar: {
      return sizeof(char);
    }
    case CollDataType::CollUint8: {
      return sizeof(uint8_t);
    }
    case CollDataType::CollInt: {
      return sizeof(int);
    }
    case CollDataType::CollUint32: {
      return sizeof(uint32_t);
    }
    case CollDataType::CollInt64: {
      return sizeof(int64_t);
    }
    case CollDataType::CollUint64: {
      return sizeof(uint64_t);
    }
    case CollDataType::CollFloat: {
      return sizeof(float);
    }
    case CollDataType::CollDouble: {
      return sizeof(double);
    }
    default: {
      LEGATE_ABORT("Unknown datatype");
      return 0;
    }
  }
}

void LocalNetwork::resetLocalBuffer(CollComm global_comm)
{
  const int global_rank                         = global_comm->global_rank;
  global_comm->local_comm->buffers[global_rank] = nullptr;
  global_comm->local_comm->displs[global_rank]  = nullptr;
}

void LocalNetwork::barrierLocal(CollComm global_comm)
{
  LegateCheck(BackendNetwork::coll_inited == true);
  const_cast<ThreadComm*>(global_comm->local_comm)->barrier_local();
}

}  // namespace legate::comm::coll
