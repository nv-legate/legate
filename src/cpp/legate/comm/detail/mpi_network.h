/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll_comm.h>
#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/mpi_interface.h>

#include <vector>

namespace legate::detail::comm::coll {

class MPINetwork final : public BackendNetwork {
  using MPIInterface = mpi::detail::MPIInterface;

 public:
  MPINetwork();

  ~MPINetwork() override;

  [[nodiscard]] int init_comm() override;

  void abort() override;

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
   * @brief Perform an all-reduce operation among the ranks of the global communicator using MPI
   * point to point communication.
   *
   * @param sendbuf The source buffer to reduce. This buffer must be of size count x CollDataType
   * size in bytes.
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

 private:
  void gather_(const void* sendbuf,
               void* recvbuf,
               int count,
               legate::comm::coll::CollDataType type,
               int root,
               legate::comm::coll::CollComm global_comm);

  /**
   * @brief Reduce the data from the send buffer to the recv buffer using MPI point to point
   * communication.
   *
   * @param sendbuf The source buffer to reduce. This buffer must be of size count x CollDataType
   * size.
   * @param recvbuf The destination buffer to receive the reduced result into. This buffer must be
   * of size count x CollDataType size.
   * @param count The number of elements to reduce.
   * @param type The data type of the elements.
   * @param op The reduction operation to perform.
   * @param root The root rank to reduce the data from.
   * @param global_comm The global communicator.
   */
  void reduce_(const void* sendbuf,
               void* recvbuf,
               int count,
               legate::comm::coll::CollDataType type,
               ReductionOpKind op,
               int root,
               legate::comm::coll::CollComm global_comm);

  void bcast_(void* buf,
              int count,
              legate::comm::coll::CollDataType type,
              int root,
              legate::comm::coll::CollComm global_comm);

  [[nodiscard]] static MPIInterface::MPI_Datatype dtype_to_mpi_dtype_(
    legate::comm::coll::CollDataType dtype);

  [[nodiscard]] int generate_alltoall_tag_(int rank1,
                                           int rank2,
                                           legate::comm::coll::CollComm global_comm) const;

  [[nodiscard]] int generate_alltoallv_tag_(int rank1,
                                            int rank2,
                                            legate::comm::coll::CollComm global_comm) const;

  [[nodiscard]] int generate_bcast_tag_(int rank, legate::comm::coll::CollComm global_comm) const;

  [[nodiscard]] int generate_gather_tag_(int rank, legate::comm::coll::CollComm global_comm) const;

  [[nodiscard]] int generate_reduce_tag_(int rank, legate::comm::coll::CollComm global_comm) const;

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

  int mpi_tag_ub_{};
  bool self_init_mpi_{};
  std::vector<MPIInterface::MPI_Comm> mpi_comms_{};
};

}  // namespace legate::detail::comm::coll
