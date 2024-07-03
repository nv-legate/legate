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

#include "core/utilities/compiler.h"
#include "core/utilities/macros.h"

#include "legate_defines.h"

#if LEGATE_DEFINED(LEGATE_USE_NETWORK) || LEGATE_DEFINED(LEGATE_DOXYGEN)
#include "core/comm/backend_network.h"
#include "core/comm/coll_comm.h"

#include <mpi.h>
#include <vector>

namespace legate::comm::coll {

class MPINetwork : public BackendNetwork {
 public:
  MPINetwork(int argc, char* argv[]);

  ~MPINetwork() override;

  [[nodiscard]] int init_comm() override;

  void abort() override;

  void comm_create(CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table) override;

  void comm_destroy(CollComm global_comm) override;

  void all_to_all_v(const void* sendbuf,
                    const int sendcounts[],
                    const int sdispls[],
                    void* recvbuf,
                    const int recvcounts[],
                    const int rdispls[],
                    CollDataType type,
                    CollComm global_comm) override;

  void all_to_all(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  CollDataType type,
                  CollComm global_comm) override;

  void all_gather(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  CollDataType type,
                  CollComm global_comm) override;

 protected:
  void gather_(const void* sendbuf,
               void* recvbuf,
               int count,
               CollDataType type,
               int root,
               CollComm global_comm);

  void bcast_(void* buf, int count, CollDataType type, int root, CollComm global_comm);

  [[nodiscard]] static MPI_Datatype dtype_to_mpi_dtype_(CollDataType dtype);

  [[nodiscard]] int generate_alltoall_tag_(int rank1, int rank2, CollComm global_comm) const;

  [[nodiscard]] int generate_alltoallv_tag_(int rank1, int rank2, CollComm global_comm) const;

  [[nodiscard]] int generate_bcast_tag_(int rank, CollComm global_comm) const;

  [[nodiscard]] int generate_gather_tag_(int rank, CollComm global_comm) const;

 private:
  int mpi_tag_ub_{};
  bool self_init_mpi_{};
  std::vector<MPI_Comm> mpi_comms_{};
};

}  // namespace legate::comm::coll

#endif
