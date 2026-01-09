/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll_comm.h>

#include <cstddef>
#include <memory>

namespace legate::detail::comm::coll {

class BackendNetwork {
 public:
  BackendNetwork()          = default;
  virtual ~BackendNetwork() = default;

  [[nodiscard]] virtual int init_comm() = 0;

  virtual void abort();

  virtual void comm_create(legate::comm::coll::CollComm global_comm,
                           int global_comm_size,
                           int global_rank,
                           int unique_id,
                           const int* mapping_table) = 0;

  virtual void comm_destroy(legate::comm::coll::CollComm global_comm) = 0;

  virtual void all_to_all_v(const void* sendbuf,
                            const int sendcounts[],
                            const int sdispls[],
                            void* recvbuf,
                            const int recvcounts[],
                            const int rdispls[],
                            legate::comm::coll::CollDataType type,
                            legate::comm::coll::CollComm global_comm) = 0;

  virtual void all_to_all(const void* sendbuf,
                          void* recvbuf,
                          int count,
                          legate::comm::coll::CollDataType type,
                          legate::comm::coll::CollComm global_comm) = 0;

  virtual void all_gather(const void* sendbuf,
                          void* recvbuf,
                          int count,
                          legate::comm::coll::CollDataType type,
                          legate::comm::coll::CollComm global_comm) = 0;

  /**
   * @brief Perform an all-reduce operation.
   *
   * @param sendbuf The source buffer to reduce. This buffer must be of size count x CollDataType
   * size.
   * @param recvbuf The destination buffer to receive the reduced result into. This buffer must be
   * of size count x CollDataType size.
   * @param count The number of elements to reduce.
   * @param type The data type of the elements.
   * @param op The reduction operation to perform.
   * @param global_comm The global communicator.
   */
  virtual void all_reduce(const void* sendbuf,
                          void* recvbuf,
                          int count,
                          legate::comm::coll::CollDataType type,
                          ReductionOpKind op,
                          legate::comm::coll::CollComm global_comm) = 0;

  static void create_network(std::unique_ptr<BackendNetwork>&& network);
  [[nodiscard]] static std::unique_ptr<BackendNetwork>& get_network();
  [[nodiscard]] static bool has_network();

  // NOLINTBEGIN(readability-identifier-naming)
  [[nodiscard]] static legate::comm::coll::CollCommType guess_comm_type_();
  // NOLINTEND(readability-identifier-naming)

 protected:
  std::int32_t get_unique_id_();

  [[nodiscard]] static void* allocate_inplace_buffer_(const void* recvbuf, std::size_t size);
  void delete_inplace_buffer_(void* buf, std::size_t size);

 public:
  legate::comm::coll::CollCommType comm_type;

 protected:
  bool coll_inited_{};
  int current_unique_id_{};
};

}  // namespace legate::detail::comm::coll
