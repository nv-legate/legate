/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll_comm.h>
#include <legate/comm/detail/backend_network.h>
#include <legate/utilities/memory.h>

#include <functional>
#include <memory>
#include <vector>

namespace legate::detail::comm::coll {

class OOBAllgather;

/**
 * @brief UCCNetwork implements the BackendNetwork interface using Unified Collective Communication
 * (UCC) library.
 * In order to initialize the UCC library, the user can provide an AllGather
 * function. This implementation uses the OOBAllgather interface so that user can provide a
 * AllGather implementation. The default implementation uses MPI for the allgather operation
 * (MPIOOBAllgather). Future work will bring in other mechanisms such as TCP/IP or third party
 * services for the allgather operation. This AllGather function is used at UCC context creation,
 * team creation, and team destruction. The UCCNetwork implements the collective operations:
 * alltoallv, alltoall, allgather, and allreduce. They use the UCC functions directly for these
 * operations.
 */
class UCCNetwork final : public BackendNetwork {
 public:
  using OOBAllgatherFactory = std::function<std::unique_ptr<OOBAllgather>(
    int rank, int size, std::vector<int> mapping_table)>;

  // Default timeout for UCC operations. Uses 30 seconds as it should be enough for timing out
  // if the operation is not completed without causing a large delay at the beginning of the
  // communicator creation.
  static constexpr std::uint32_t DEFAULT_TIMEOUT_SECONDS = 30;

  /**
   * @brief Constructor for UCCNetwork
   *
   * @param oob_factory Factory function for creating OOBAllgather instances. The OOBAllGather
   * is used for out-of-band allgather operation. The factory function is used to create the
   * OOBAllgather instance for each rank.
   * @param timeout Timeout for UCC operations. This is used in team creation and destruction.
   */
  explicit UCCNetwork(OOBAllgatherFactory oob_factory,
                      std::uint32_t timeout = DEFAULT_TIMEOUT_SECONDS);

  /**
   * @brief Constructor for UCCNetwork with default MPI OOB factory
   *
   * @param timeout Timeout for UCC operations. This is used in team creation and destruction.
   */
  explicit UCCNetwork(std::uint32_t timeout = DEFAULT_TIMEOUT_SECONDS);

  ~UCCNetwork() noexcept override;

  // Non-copyable, movable
  UCCNetwork(const UCCNetwork&)                = delete;
  UCCNetwork& operator=(const UCCNetwork&)     = delete;
  UCCNetwork(UCCNetwork&&) noexcept            = default;
  UCCNetwork& operator=(UCCNetwork&&) noexcept = default;

  /**
   * @brief Initialize the UCCNetwork.
   *
   * @return Unique identifier for the communicator
   */
  [[nodiscard]] int init_comm() override;

  /**
   * @brief This method shuts down the UCCNetwork.
   */
  void abort() override;

  /**
   * @brief Create a communicator. This will create a ucc_context and a ucc_team for each
   * rank in the global communicator. The mapping table is used for mapping each global rank
   * to the corresponding process rank.
   *
   * @param global_comm Global communicator
   * @param global_comm_size Size of the global communicator
   * @param global_rank Rank of the current process in the global communicator
   * @param unique_id Unique identifier for the communicator
   * @param mapping_table Mapping table for the communicator
   */
  void comm_create(legate::comm::coll::CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table) override;

  /**
   * @brief Destroy the communicator
   *
   * @param global_comm Global communicator
   */
  void comm_destroy(legate::comm::coll::CollComm global_comm) override;

  /**
   * @brief Perform alltoallv operation using UCC. Each rank sends data to all other ranks according
   * to the sendcounts and send offsets. The receive buffer is a single buffer that is used to
   * receive data from all ranks. The receives are placed in the receive buffer according to the
   * receive displacements.
   *
   * @param sendbuf Input buffer containing data to be gathered from this rank
   * @param sendcounts Number of elements to send from this rank
   * @param sdispls Displacements of the send buffer for each rank
   * @param recvbuf Output buffer to receive gathered data from all ranks
   * @param recvcounts Number of elements to receive from each rank (must be same as sendcounts)
   * @param rdispls Displacements of the receive data from each rank into the receive buffer
   * @param type Data type of the elements
   * @param global_comm Global communicator
   */
  void all_to_all_v(const void* sendbuf,
                    const int sendcounts[],
                    const int sdispls[],
                    void* recvbuf,
                    const int recvcounts[],
                    const int rdispls[],
                    legate::comm::coll::CollDataType type,
                    legate::comm::coll::CollComm global_comm) override;

  /**
   * @brief Perform alltoall operation using UCC. All the ranks exchange data with
   * all other ranks.
   *
   * @param sendbuf Input buffer containing data to be gathered from this rank
   * @param recvbuf Output buffer to receive gathered data from all ranks
   * @param count Number of elements to gather
   * @param type Data type of the elements
   * @param global_comm Global communicator
   */
  void all_to_all(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  legate::comm::coll::CollComm global_comm) override;

  /**
   * @brief Perform allgather operation using UCC. This gathers data from all ranks into a single
   * buffer in all ranks.
   *
   * @param sendbuf Input buffer containing data to be gathered from this rank
   * @param recvbuf Output buffer to receive gathered data from all ranks
   * @param count Number of elements to gather
   * @param type Data type of the elements
   * @param global_comm Global communicator
   */
  void all_gather(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  legate::comm::coll::CollComm global_comm) override;

  /**
   * @brief Perform allreduce operation using UCC. This performs a reduction operation across all
   * ranks and distributes the result to all ranks.
   *
   * @param sendbuf Input buffer containing data to be reduced from this rank.
   * @param recvbuf Output buffer to receive reduced data.
   * @param count Number of elements to reduce.
   * @param type Data type of the elements.
   * @param op Reduction operation to perform.
   * @param global_comm Global communicator.
   */
  void all_reduce(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  ReductionOpKind op,
                  legate::comm::coll::CollComm global_comm) override;

  /**
   * @brief Shutdown the UCCNetwork
   */
  void shutdown();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_{};
};

}  // namespace legate::detail::comm::coll
