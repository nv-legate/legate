/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

 protected:
  [[nodiscard]] static std::size_t get_dtype_size_(legate::comm::coll::CollDataType dtype);

  void reset_local_buffer_(legate::comm::coll::CollComm global_comm);

  void barrier_local_(legate::comm::coll::CollComm global_comm);

 private:
  std::vector<std::unique_ptr<ThreadComm>> thread_comms_{};
};

}  // namespace legate::detail::comm::coll
