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

#include "core/comm/coll_comm.h"

#include <cstddef>
#include <memory>

namespace legate::comm::coll {

class BackendNetwork {
 public:
  BackendNetwork()          = default;
  virtual ~BackendNetwork() = default;

  [[nodiscard]] virtual int init_comm() = 0;

  virtual void abort();

  [[nodiscard]] virtual CollStatus comm_create(CollComm global_comm,
                                               int global_comm_size,
                                               int global_rank,
                                               int unique_id,
                                               const int* mapping_table) = 0;

  [[nodiscard]] virtual CollStatus comm_destroy(CollComm global_comm) = 0;

  [[nodiscard]] virtual CollStatus all_to_all_v(const void* sendbuf,
                                                const int sendcounts[],
                                                const int sdispls[],
                                                void* recvbuf,
                                                const int recvcounts[],
                                                const int rdispls[],
                                                CollDataType type,
                                                CollComm global_comm) = 0;

  [[nodiscard]] virtual CollStatus all_to_all(
    const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm) = 0;

  [[nodiscard]] virtual CollStatus all_gather(
    const void* sendbuf, void* recvbuf, int count, CollDataType type, CollComm global_comm) = 0;

 protected:
  [[nodiscard]] CollStatus get_unique_id_(int* id);

  [[nodiscard]] static void* allocate_inplace_buffer_(const void* recvbuf, std::size_t size);
  void delete_inplace_buffer_(void* buf, std::size_t size);

 public:
  CollCommType comm_type;

 protected:
  bool coll_inited_{};
  int current_unique_id_{};
};

extern std::unique_ptr<BackendNetwork> backend_network;

}  // namespace legate::comm::coll
