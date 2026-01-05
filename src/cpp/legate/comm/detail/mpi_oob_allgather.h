/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/detail/oob_allgather.h>

#include <ucc/api/ucc.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace legate::detail::comm::coll {

/**
 * @brief MPI implementation for OOBAllgather. This implements a a blocking allgather
 * using MPI point-to-point communications. The implementation first gathers the data
 * from all ranks to the root rank, and then broadcasts the data from the root rank to all ranks.
 */
class MPIOOBAllgather final : public OOBAllgather {
 public:
  /**
   * @brief Constructor for MPI OOB allgather
   *
   * @param rank Global rank of the task/thread used by Legate
   * @param size Number of parallel tasks/threads used by Legate
   * @param mapping_table Mapping table for the communicator
   */
  MPIOOBAllgather(int rank, int size, std::vector<int> mapping_table);
  ~MPIOOBAllgather() override;

  MPIOOBAllgather(const MPIOOBAllgather&)            = delete;
  MPIOOBAllgather& operator=(const MPIOOBAllgather&) = delete;
  MPIOOBAllgather(MPIOOBAllgather&&)                 = delete;
  MPIOOBAllgather& operator=(MPIOOBAllgather&&)      = delete;

  /**
   * @brief Perform out-of-band allgather operation using MPI point-to-point communication.
   * This implementation uses a blocking allgather. First it gathers the data from all ranks to the
   * root rank, and then broadcasts the data from the root rank to all ranks. Because it is a
   * blocking allgather, the request object is not used.
   *
   * @param sendbuf Input buffer containing data to be gathered from this rank
   * @param recvbuf Output buffer to receive gathered data from all ranks
   * @param message_size Size in bytes of the message to send
   * @param allgather_info Pointer to the OOBAllgather instance
   * @param request Pointer to the request object
   * @return UCC status, UCC_OK if successful, UCC_ERR_NO_MESSAGE if failed
   */
  [[nodiscard]] ucc_status_t allgather(const void* sendbuf,
                                       void* recvbuf,
                                       std::size_t message_size,
                                       void* allgather_info,
                                       void** request) override;

  /**
   * @brief Test if the request is completed. This implementation uses a blocking allgather.
   * So test and free just returns UCC_OK without doing anything.
   *
   * @param request Pointer to the request object
   * @return UCC status, UCC_OK if successful, UCC_ERR_NO_MESSAGE if failed
   */
  [[nodiscard]] ucc_status_t test(void* request) override;

  /**
   * @brief Free the request object. This implementation uses a blocking allgather. So test and
   * free just returns UCC_OK without doing anything.

   * @param request Pointer to the request object
   * @return UCC status, UCC_OK if successful, UCC_ERR_NO_MESSAGE if failed
   */
  [[nodiscard]] ucc_status_t free(void* request) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_{};
};

/**
 * @brief Factory function that returns a default MPIOOBAllgather factory
 * @return Factory function that creates MPIOOBAllgather instances
 */
[[nodiscard]] std::function<std::unique_ptr<OOBAllgather>(int, int, std::vector<int>)>
create_mpi_oob_allgather_factory();

}  // namespace legate::detail::comm::coll
