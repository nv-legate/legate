/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/mpi_oob_allgather.h>

#include <legate_defines.h>

#include <legate/comm/detail/logger.h>
#include <legate/comm/detail/mpi_interface.h>
#include <legate/comm/detail/mpi_network.h>
#include <legate/comm/detail/oob_allgather.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/scope_guard.h>

#include <ucc/api/ucc.h>

#include <fmt/format.h>

#include <cstring>
#include <stdexcept>

namespace legate::detail::comm::coll {

//  MPI error checking that returns UCC_ERR_NO_MESSAGE
#define CHECK_MPI_AND_RETURN(...)                                                          \
  do {                                                                                     \
    const int lgcore_check_mpi_result_ = __VA_ARGS__;                                      \
    if (LEGATE_UNLIKELY(lgcore_check_mpi_result_ !=                                        \
                        legate::detail::comm::mpi::detail::MPIInterface::MPI_SUCCESS())) { \
      logger().debug() << fmt::format("MPI error: {} in {}:{}",                            \
                                      lgcore_check_mpi_result_,                            \
                                      LEGATE_STRINGIZE(__FILE__),                          \
                                      LEGATE_STRINGIZE(__LINE__));                         \
      return UCC_ERR_NO_MESSAGE;                                                           \
    }                                                                                      \
  } while (0)

using MPIInterface = legate::detail::comm::mpi::detail::MPIInterface;

// MPI tag generation constants
namespace {

constexpr int GATHER_TAG_MAX    = 100;
constexpr int GATHER_TAG_OFFSET = 4;
constexpr int BCAST_TAG_MAX     = 10;
constexpr int BCAST_TAG_OFFSET  = 0;

}  // namespace

class MPIOOBAllgather::Impl {
 public:
  /**
   * @brief Constructor for MPIOOBAllgather::Impl
   *
   * @param rank Rank of the current process
   * @param size Size of the global communicator
   * @param mapping_table Mapping table for the communicator
   */
  Impl(int rank, int size, std::vector<int> mapping_table);
  ~Impl();

  [[nodiscard]] ucc_status_t allgather(const void* sendbuf,
                                       void* recvbuf,
                                       std::size_t message_size,
                                       void* allgather_info,
                                       void** request);
  [[nodiscard]] ucc_status_t test(void* request);
  [[nodiscard]] ucc_status_t free(void* request);

 private:
  [[nodiscard]] int generate_tag_(int rank, int max, int offset) const;
  [[nodiscard]] ucc_status_t gather_(const void* sendbuf,
                                     void* recvbuf,
                                     std::size_t count,
                                     int root);
  [[nodiscard]] ucc_status_t bcast_(void* buf, std::size_t count, int root);

  bool mpi_initialized_{false};
  int mpi_rank_{-1};
  int mpi_size_{-1};
  int global_rank_{-1};
  int global_size_{-1};
  // MPI tag upper bound
  int mpi_tag_ub_{-1};
  // MPI type extent for MPI_UINT8_T that is used for allgather, bcast and gather
  MPIInterface::MPI_Aint type_extent_{};
  // This is a duplicate of MPI_COMM_WORLD. It is to isolate the communication within this
  // class from rest of the applications use of MPI.
  MPIInterface::MPI_Comm comm_{};
  // Mapping table from global rank to MPI rank (global_rank -> mpi_rank)
  std::vector<int> mapping_table_{};
};

MPIOOBAllgather::Impl::Impl(int rank, int size, std::vector<int> mapping_table)
  : global_rank_{rank}, global_size_{size}, mapping_table_{std::move(mapping_table)}
{
  // Check if MPI is already initialized
  int init_flag = 0;

  LEGATE_CHECK_MPI(MPIInterface::mpi_initialized(&init_flag));
  if (!init_flag) {
    LEGATE_ABORT("MPI not available for MPIOOBAllgather, rank: {}, size: {}", rank, size);
  }

  int mpi_thread_model;

  LEGATE_CHECK_MPI(MPIInterface::mpi_query_thread(&mpi_thread_model));
  if (mpi_thread_model != MPIInterface::MPI_THREAD_MULTIPLE()) {
    LEGATE_ABORT("MPI has been initialized without MPI_THREAD_MULTIPLE");
  }

  // Duplicate MPI_COMM_WORLD for our use, this is to isolate the communication within this
  // class from rest of the applications use of MPI.
  LEGATE_CHECK_MPI(MPIInterface::mpi_comm_dup(MPIInterface::MPI_COMM_WORLD(), &comm_));
  LEGATE_CHECK_MPI(MPIInterface::mpi_comm_rank(comm_, &mpi_rank_));
  LEGATE_CHECK_MPI(MPIInterface::mpi_comm_size(comm_, &mpi_size_));

  int flag = 0;
  // tag_ub is clearly modified
  int* tag_ub = nullptr;  // NOLINT(misc-const-correctness)

  // Get the upper bound of tag values
  LEGATE_CHECK_MPI(MPIInterface::mpi_comm_get_attr(MPIInterface::MPI_COMM_WORLD(),
                                                   MPIInterface::MPI_TAG_UB(),
                                                   static_cast<void*>(&tag_ub),
                                                   &flag));

  const auto mpi_type = MPIInterface::MPI_UINT8_T();
  MPIInterface::MPI_Aint lb{};

  LEGATE_CHECK_MPI(MPIInterface::mpi_type_get_extent(mpi_type, &lb, &type_extent_));

  mpi_tag_ub_      = *tag_ub;
  mpi_initialized_ = true;

  logger().debug() << fmt::format(
    "MPIOOBAllgather initialized with MPI (rank={}, size={})", mpi_rank_, mpi_size_);
}

MPIOOBAllgather::Impl::~Impl()
{
  if (mpi_initialized_) {
    static_cast<void>(MPIInterface::mpi_comm_free(&comm_));
  }
}

ucc_status_t MPIOOBAllgather::Impl::allgather(const void* sendbuf,
                                              void* recvbuf,
                                              std::size_t message_size,
                                              void* /*allgather_info*/,
                                              void** /*request*/)
{
  auto* sendbuf_tmp    = const_cast<void*>(sendbuf);
  const auto num_bytes = static_cast<std::size_t>(type_extent_ * message_size);
  std::unique_ptr<char[]> sendbuf_tmp_unique;

  // both sendbuf and recvbuf are the same, we need to make a copy of the sendbuf
  // as we are doing a gather and bcast.
  if (sendbuf == recvbuf) {
    LEGATE_ASSERT(num_bytes);
    sendbuf_tmp_unique = std::make_unique<char[]>(num_bytes);
    std::memcpy(sendbuf_tmp_unique.get(), recvbuf, num_bytes);
    sendbuf_tmp = sendbuf_tmp_unique.get();
  }

  const int root = 0;
  // Gather from all ranks to root rank (0)
  if (const ucc_status_t status = gather_(sendbuf_tmp, recvbuf, message_size, root);
      status != UCC_OK) {
    return status;
  }

  // Broadcast from root rank (0) to all ranks
  if (const ucc_status_t status = bcast_(recvbuf, message_size * global_size_, root);
      status != UCC_OK) {
    return status;
  }

  return UCC_OK;
}

ucc_status_t MPIOOBAllgather::Impl::gather_(const void* sendbuf,
                                            void* recvbuf,
                                            std::size_t count,
                                            int root)
{
  // Should not see inplace here
  LEGATE_CHECK(sendbuf != recvbuf);

  const auto mpi_type     = MPIInterface::MPI_UINT8_T();
  const int root_mpi_rank = mapping_table_[root];

  // root receives from all other ranks
  if (global_rank_ == root) {
    const auto size = static_cast<std::size_t>(type_extent_ * static_cast<std::ptrdiff_t>(count));
    auto* dst       = static_cast<char*>(recvbuf);

    LEGATE_CHECK(dst != nullptr);

    for (int i = 0; i < global_size_; i++) {
      const auto recvfrom_mpi_rank = mapping_table_[i];
      const auto tag               = generate_tag_(i, GATHER_TAG_MAX, GATHER_TAG_OFFSET);

      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        logger().debug() << fmt::format(
          "GatherMPI: root recv global_rank {}, mpi rank {}, recv from {} ({}), tag {}",
          global_rank_,
          mpi_rank_,
          i,
          recvfrom_mpi_rank,
          tag);
      }
      if (global_rank_ == i) {
        std::memcpy(dst, sendbuf, size);
      } else {
        MPIInterface::MPI_Status status;

        CHECK_MPI_AND_RETURN(MPIInterface::mpi_recv(
          dst, count, mpi_type, recvfrom_mpi_rank, tag, MPIInterface::MPI_COMM_WORLD(), &status));
      }
      dst += size;
    }
    return UCC_OK;
  }

  // non-root ranks sends to root
  const auto tag = generate_tag_(global_rank_, GATHER_TAG_MAX, GATHER_TAG_OFFSET);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    logger().debug() << fmt::format(
      "GatherMPI: non-root send global_rank {}, mpi rank {}, send to {} ({}), tag {}",
      global_rank_,
      mpi_rank_,
      root,
      root_mpi_rank,
      tag);
  }
  CHECK_MPI_AND_RETURN(MPIInterface::mpi_send(
    sendbuf, count, mpi_type, root_mpi_rank, tag, MPIInterface::MPI_COMM_WORLD()));
  return UCC_OK;
}

ucc_status_t MPIOOBAllgather::Impl::bcast_(void* buf, std::size_t count, int root)
{
  const auto mpi_type = MPIInterface::MPI_UINT8_T();
  // root rank sends to all other ranks
  if (global_rank_ == root) {
    for (int i = 0; i < global_size_; i++) {
      const auto sendto_mpi_rank = mapping_table_[i];
      const auto tag             = generate_tag_(i, BCAST_TAG_MAX, BCAST_TAG_OFFSET);

      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        logger().debug() << fmt::format(
          "BcastMPI: root send global_rank {}, mpi rank {}, send to {} ({}), tag {}",
          global_rank_,
          mpi_rank_,
          i,
          sendto_mpi_rank,
          tag);
      }
      if (global_rank_ != i) {
        CHECK_MPI_AND_RETURN(MPIInterface::mpi_send(
          buf, count, mpi_type, sendto_mpi_rank, tag, MPIInterface::MPI_COMM_WORLD()));
      }
    }

    return UCC_OK;
  }

  // non-root ranks receive from root rank
  const int root_mpi_rank = mapping_table_[root];
  const auto tag          = generate_tag_(global_rank_, BCAST_TAG_MAX, BCAST_TAG_OFFSET);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    logger().debug() << fmt::format(
      "BcastMPI: non-root recv global_rank {}, mpi rank {}, recv from {} ({}), tag {}",
      global_rank_,
      mpi_rank_,
      root,
      root_mpi_rank,
      tag);
  }

  MPIInterface::MPI_Status status;

  CHECK_MPI_AND_RETURN(MPIInterface::mpi_recv(
    buf, count, mpi_type, root_mpi_rank, tag, MPIInterface::MPI_COMM_WORLD(), &status));
  return UCC_OK;
}

int MPIOOBAllgather::Impl::generate_tag_(int rank, int max, int offset) const
{
  const int tag = (rank * max) + offset;

  LEGATE_CHECK(tag <= mpi_tag_ub_ && tag >= 0);
  return tag;
}

ucc_status_t MPIOOBAllgather::Impl::test(void* /*request*/) { return UCC_OK; }

ucc_status_t MPIOOBAllgather::Impl::free(void* /*request*/) { return UCC_OK; }

MPIOOBAllgather::MPIOOBAllgather(int rank, int size, std::vector<int> mapping_table)
  : impl_{std::make_unique<Impl>(rank, size, std::move(mapping_table))}
{
}

MPIOOBAllgather::~MPIOOBAllgather() = default;

ucc_status_t MPIOOBAllgather::allgather(const void* sendbuf,
                                        void* recvbuf,
                                        std::size_t message_size,
                                        void* allgather_info,
                                        void** request)
{
  return impl_->allgather(sendbuf, recvbuf, message_size, allgather_info, request);
}

ucc_status_t MPIOOBAllgather::test(void* request) { return impl_->test(request); }

ucc_status_t MPIOOBAllgather::free(void* request) { return impl_->free(request); }

std::function<std::unique_ptr<OOBAllgather>(int, int, std::vector<int>)>
create_mpi_oob_allgather_factory()
{
  return [](int rank, int size, std::vector<int> mapping_table) -> std::unique_ptr<OOBAllgather> {
    return std::make_unique<MPIOOBAllgather>(rank, size, std::move(mapping_table));
  };
}

}  // namespace legate::detail::comm::coll
