/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/thread_comm.h>

#include <memory>
#include <vector>

namespace legate::detail::comm::coll {

class LocalNetwork : public BackendNetwork {
 public:
  LocalNetwork();

  ~LocalNetwork() override;

  [[nodiscard]] int init_comm() override;

  void comm_create(legate::comm::coll::CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table) override;

  void comm_destroy(legate::comm::coll::CollComm global_comm) override;

  void all_to_all_v(const void* sendbuf,
                    const int sendcounts[],
                    const int sdispls[],
                    void* recvbuf,
                    const int recvcounts[],
                    const int rdispls[],
                    legate::comm::coll::CollDataType type,
                    legate::comm::coll::CollComm global_comm) override;

  void all_to_all(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  legate::comm::coll::CollComm global_comm) override;

  void all_gather(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  legate::comm::coll::CollComm global_comm) override;

  /**
   * @brief Perform an all-reduce operation among the ranks of the global communicator using local
   * memory reductions.
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
  void all_reduce(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  ReductionOpKind op,
                  legate::comm::coll::CollComm global_comm) override;

 protected:
  [[nodiscard]] static std::size_t get_dtype_size_(legate::comm::coll::CollDataType dtype);

  /** @brief Apply a reduction operation for each index in destination and source buffer. Store
   * result in destination buffer.
   *
   * @param dst Destination buffer (also serves as one input, modified in-place).
   * @param src Source buffer.
   * @param count Number of elements.
   * @param op Reduction operation to apply.
   */
  static void apply_reduction_(void* dst,
                               const void* src,
                               int count,
                               legate::comm::coll::CollDataType type,
                               ReductionOpKind op);

  void reset_local_buffer_(legate::comm::coll::CollComm global_comm);

  void barrier_local_(legate::comm::coll::CollComm global_comm);

 private:
  std::vector<std::unique_ptr<ThreadComm>> thread_comms_{};
};

}  // namespace legate::detail::comm::coll
