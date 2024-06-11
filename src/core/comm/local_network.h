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

#pragma once

#include "core/comm/backend_network.h"

#include <memory>
#include <vector>

namespace legate::comm::coll {

class LocalNetwork : public BackendNetwork {
 public:
  LocalNetwork(int argc, char* argv[]);

  ~LocalNetwork() override;

  [[nodiscard]] int init_comm() override;

  [[nodiscard]] CollStatus comm_create(CollComm global_comm,
                                       int global_comm_size,
                                       int global_rank,
                                       int unique_id,
                                       const int* mapping_table) override;

  [[nodiscard]] CollStatus comm_destroy(CollComm global_comm) override;

  [[nodiscard]] CollStatus all_to_all_v(const void* sendbuf,
                                        const int sendcounts[],
                                        const int sdispls[],
                                        void* recvbuf,
                                        const int recvcounts[],
                                        const int rdispls[],
                                        CollDataType type,
                                        CollComm global_comm) override;

  [[nodiscard]] CollStatus all_to_all(const void* sendbuf,
                                      void* recvbuf,
                                      int count,
                                      CollDataType type,
                                      CollComm global_comm) override;

  [[nodiscard]] CollStatus all_gather(const void* sendbuf,
                                      void* recvbuf,
                                      int count,
                                      CollDataType type,
                                      CollComm global_comm) override;

 protected:
  [[nodiscard]] static std::size_t get_dtype_size_(CollDataType dtype);

  void reset_local_buffer_(CollComm global_comm);

  void barrier_local_(CollComm global_comm);

 private:
  std::vector<std::unique_ptr<ThreadComm>> thread_comms_{};
};

}  // namespace legate::comm::coll
