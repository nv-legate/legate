/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/ucc_network.h>

#include <legate_defines.h>

#include <legate/comm/detail/logger.h>
#include <legate/comm/detail/mpi_oob_allgather.h>
#include <legate/comm/detail/oob_allgather.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/scope_guard.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <ucc/api/ucc.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate::detail::comm::coll {

// UCC error checking macro to check the UCC call result and abort the program if the UCC call
// fails.
#define LEGATE_CHECK_UCC(...)                                                               \
  do {                                                                                      \
    const int lgcore_check_ucc_result_ = __VA_ARGS__;                                       \
    if (LEGATE_UNLIKELY(lgcore_check_ucc_result_ != UCC_OK)) {                              \
      LEGATE_ABORT("Internal UCC failure with error code ",                                 \
                   lgcore_check_ucc_result_,                                                \
                   " in " LEGATE_STRINGIZE(__FILE__) ":" LEGATE_STRINGIZE(__LINE__) " in ", \
                   __func__,                                                                \
                   "(): " LEGATE_STRINGIZE(__VA_ARGS__));                                   \
    }                                                                                       \
  } while (0)

/**
 * @brief Holds the UCC context, team, and other information for a communicator.
 *
 * This will be created for each comm_create call in UCCNetwork::Impl and destroyed when
 * comm_destroy is called. The destructor doesn't destroy the UCC context and team, it is the
 * responsibility of the UCCNetwork::Impl to do so. This is done to avoid deadlock when destroying
 * the UCC context and team.
 */
class UCCCommunicator {
 public:
  /**
   * @brief Creates a UCCCommunicator
   *
   * @param global_rank The global rank of the communicator
   * @param global_size The global size of the communicator
   * @param id The unique ID of the communicator
   * @param mapping The mapping table from global rank to process rank
   * @param allgather The OOBAllgather to be used for communicator creation and destruction
   * @param ucc_context The UCC context for the communicator
   * @param ucc_team The UCC team for the communicator
   */
  UCCCommunicator(int global_rank,
                  std::size_t global_size,
                  std::unique_ptr<OOBAllgather> allgather,
                  std::uint32_t timeout,
                  ucc_lib_h lib);

  /**
   * @brief Destroy UCC team and context
   */
  void destroy_ucc_team_and_context();

  /**
   * @brief Get the size of the communicator
   */
  [[nodiscard]] std::size_t get_size() const;

  /**
   * @brief Post the UCC collective operation and wait for completion
   *
   * @param coll_args The UCC collective arguments
   *
   * @return The UCC status UCC_OK if the collective operation is successful.
   */
  [[nodiscard]] ucc_status_t ucc_collective(ucc_coll_args_t* coll_args);

 private:
  /**
   * @brief Create UCC context for the communicator. This function will abort the program if the
   * UCC context creation fails.
   *
   * @param rank The global rank of the communicator
   * @param size The size of the communicator
   * @param lib The UCC library handle
   * @param oob_allgather The OOBAllgather to be used for communicator creation and destruction
   *
   * @return The UCC context
   */
  [[nodiscard]] static ucc_context_h create_ucc_ctx_(int rank,
                                                     std::size_t size,
                                                     ucc_lib_h lib,
                                                     OOBAllgather* oob_allgather);

  /**
   * @brief Create UCC team for the communicator. This function will abort the program if the
   * UCC team creation fails.
   *
   * @param rank The global rank of the communicator
   * @param size The global size of the communicator
   * @param ctx The UCC context
   * @param oob_allgather The OOBAllgather to be used for communicator creation and destruction
   * @param timeout The timeout for the UCC team creation
   *
   * @return The UCC team
   */
  [[nodiscard]] static ucc_team_h create_ucc_team_(int rank,
                                                   std::size_t size,
                                                   ucc_context_h ctx,
                                                   OOBAllgather* oob_allgather,
                                                   std::uint32_t timeout);

  std::size_t size_{};
  // both these should be not null once the UCC communicator is created.
  // otherwise the creation will fail.
  ucc_context_h context_{nullptr};
  ucc_team_h team_{nullptr};
  // Out-of-band allgather use for UCC bootstrap and destruction.
  std::unique_ptr<OOBAllgather> oob_allgather_{nullptr};
  std::size_t timeout_{UCCNetwork::DEFAULT_TIMEOUT_SECONDS};
};

class UCCNetwork::Impl {
 public:
  /**
   * @brief Constructor for UCCNetwork::Impl
   *
   * @param oob_factory Factory function for creating OOBAllgather instances. The OOBAllGather
   * is used for out-of-band allgather operation. The factory function is used to create the
   * OOBAllgather instance for each rank.
   * @param timeout Timeout for UCC operations. This is used in team creation and destruction.
   */
  explicit Impl(UCCNetwork::OOBAllgatherFactory oob_factory, std::uint32_t timeout);
  ~Impl();

  // Non-copyable, non-movable
  Impl(const Impl&)            = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&)                 = delete;
  Impl& operator=(Impl&&)      = delete;

  /**
   * @brief Initialize the UCC network

   * @param id The unique ID of the communicator

   * @return The unique ID of the communicator
   */
  [[nodiscard]] int init_comm(int id);

  /**
   * @brief Shutdown the UCC network.
   */
  void abort();

  /**
   * @brief Destroy all the UCC communicators (contexts and teams) and finalize the UCC library.
   * If UUC team cannot be shutdown within the timeout, this function will return.
   */
  void shutdown();

  /**
   * @brief Create a the UCC communicator. This function will create the UCC context and team. For
   * each thread, we create a UCC communicator.
   *
   * @param global_comm The global communicator, this holds the unique id.
   * @param global_comm_size Number of threads / processes in the communicator
   * @param global_rank The global rank of this communicator.
   * @param unique_id The unique ID of the communicator. This will be used by subsequent calls to
   * identify the communicator.
   * @param mapping_table The mapping table from global rank to process rank
   */
  void comm_create(legate::comm::coll::CollComm global_comm,
                   int global_comm_size,
                   int global_rank,
                   int unique_id,
                   const int* mapping_table);

  /**
   * @brief Destroy the UCC context and team associated with the unique id.
   *
   * @param global_comm The global communicator, this holds the unique id.
   */
  void comm_destroy(legate::comm::coll::CollComm global_comm);

  /**
   * @brief Perform an all-to-all-v operation.
   *
   * @param sendbuf The buffer to send from
   * @param sendcounts The number of elements to send to each rank
   * @param sdispls The displacement the elements in the send buffer
   * @param recvbuf The buffer to receive into. This buffer must be able to hold all the elements
   * received from all ranks.
   * @param recvcounts The number of elements to receive from each rank
   * @param rdispls The offset into the receive buffer to receive from each rank
   * @param type The data type of the elements
   * @param global_comm The global communicator, this holds the unique id.
   */
  void all_to_all_v(const void* sendbuf,
                    const int sendcounts[],
                    const int sdispls[],
                    void* recvbuf,
                    const int recvcounts[],
                    const int rdispls[],
                    legate::comm::coll::CollDataType type,
                    legate::comm::coll::CollComm global_comm);

  /**
   * @brief Perform an all-to-all operation.
   *
   * @param sendbuf The buffer to send from
   * @param recvbuf The buffer to receive into. This buffer must be of size global_comm_size x count
   * x dtype_size.
   * @param count The number of elements to send
   * @param type The data type of the elements
   * @param global_comm The global communicator, this holds the unique id.
   */
  void all_to_all(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  legate::comm::coll::CollComm global_comm);

  /**
   * @brief Perform an all-gather operation.
   *
   * @param sendbuf The buffer to send
   * @param recvbuf The buffer to receive into. This buffer must be of size global_comm_size x count
   * x dtype_size.
   * @param count The number of elements to send
   * @param type The data type of the elements
   * @param global_comm The global communicator, this holds the unique id.
   */
  void all_gather(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  legate::comm::coll::CollComm global_comm);

  /**
   * @brief Perform an all-reduce operation.
   *
   * @param sendbuf The buffer to reduce from
   * @param recvbuf The buffer to receive the reduced result into. This buffer must be of size count
   * x dtype_size.
   * @param count The number of elements to reduce
   * @param type The data type of the elements
   * @param op The reduction operation to perform
   * @param global_comm The global communicator, this holds the unique id.
   */
  void all_reduce(const void* sendbuf,
                  void* recvbuf,
                  int count,
                  legate::comm::coll::CollDataType type,
                  ReductionOpKind op,
                  legate::comm::coll::CollComm global_comm);

 private:
  /**
   * @brief Initialize UCC library, this will call ucc_init() and abort the program if the UCC
   * library initialization fails.
   */
  void init_ucc_lib_();

  /**
   * @brief Get the UCC data type from the Legate data type
   *
   * @param dtype The Legate data type

   * @return The UCC data type
   */
  [[nodiscard]] static ucc_datatype_t dtype_to_ucc_dtype_(legate::comm::coll::CollDataType dtype);

  /**
   * @brief Get the UCC reduction operation from the Legate reduction operation
   *
   * @param op The Legate reduction operation
   *
   * @return The UCC reduction operation
   */
  [[nodiscard]] static ucc_reduction_op_t redop_to_ucc_redop_(ReductionOpKind op);

  /**
   * @brief Get the size of the UCC data type
   *
   * @param dtype The Legate data type

   * @return The size of the UCC data type
   */
  [[nodiscard]] static std::size_t get_dtype_size_(legate::comm::coll::CollDataType dtype);

  /**
   * @brief Make a UCC collectives arguments for the given parameters.
   *
   * @param sendbuf The buffer to send from.
   * @param recvbuf The buffer to receive into.
   * @param send_count The number of elements to send.
   * @param recv_count The number of elements to receive.
   * @param recv_count The number of elements to receive.
   * @param coll_type The type of collective to perform.
   * @param type The data type of the elements.
   *
   * @return The UCC collectives arguments.
   */
  [[nodiscard]] ucc_coll_args_t make_ucc_coll_args_(const void* sendbuf,
                                                    void* recvbuf,
                                                    std::uint64_t send_count,
                                                    std::uint64_t recv_count,
                                                    ucc_coll_type_t coll_type,
                                                    legate::comm::coll::CollDataType type);

  // This is the UCC library handle, it is initialized when the UCCNetwork is created.
  // this should be available for the entire lifetime of the UCCNetwork.
  std::optional<ucc_lib_h> lib_{};
  // Timeout for UCC team creation and destruction
  std::uint32_t timeout_{DEFAULT_TIMEOUT_SECONDS};

  // Communicator management, global_rank -> UCCCommunicator
  std::unordered_map<int, std::unique_ptr<UCCCommunicator>> ucc_comms_{};

  // OOBAllgather factory function
  UCCNetwork::OOBAllgatherFactory oob_factory_{nullptr};
  // lock for accessing ucc_comms_
  std::mutex ucc_comms_lock_{};
};

// Implementation of UCCNetwork::Impl
UCCNetwork::Impl::Impl(UCCNetwork::OOBAllgatherFactory oob_factory, std::uint32_t timeout)
  : timeout_{timeout}, oob_factory_{std::move(oob_factory)}
{
}

UCCNetwork::Impl::~Impl() { shutdown(); }

void UCCNetwork::Impl::shutdown()
{
  if (!lib_.has_value()) {
    return;
  }

  std::vector<std::thread> threads;
  // We need to use threads to avoid deadlock when destroying.
  // All the context/team destructions need to progress in parallel, otherwise
  // they cannot proceed as they are using the oob_allgather which needs
  // all of them to participate.
  threads.reserve(ucc_comms_.size());
  for (auto& ucc_comm : ucc_comms_) {
    threads.emplace_back(
      [ucc_comm_ptr = ucc_comm.second.get()]() { ucc_comm_ptr->destroy_ucc_team_and_context(); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  ucc_comms_.clear();

  if (lib_.has_value()) {
    ucc_finalize(lib_.value());
    lib_.reset();
  }
}

int UCCNetwork::Impl::init_comm(int id)
{
  // Initialize UCC library
  init_ucc_lib_();
  LEGATE_CHECK(ucc_comms_.empty());

  return id;
}

void UCCNetwork::Impl::abort()
{
  // For UCC, we don't have a direct abort mechanism like MPI_Abort
  shutdown();
}

void UCCNetwork::Impl::comm_create(legate::comm::coll::CollComm global_comm,
                                   int global_comm_size,
                                   int global_rank,
                                   int unique_id,
                                   const int* mapping_table)
{
  LEGATE_CHECK(lib_.has_value());

  {
    const std::scoped_lock<std::mutex> lock{ucc_comms_lock_};

    if (ucc_comms_.find(global_rank) != ucc_comms_.end()) {
      LEGATE_ABORT(
        fmt::format("Communicator already exists for this global rank: {}", global_rank));
    }
  }

  LEGATE_CHECK(global_comm_size > 0);
  LEGATE_CHECK(global_rank >= 0);
  LEGATE_CHECK(global_rank < global_comm_size);
  LEGATE_CHECK(mapping_table != nullptr);

  std::vector<int> mapping_table_vec(mapping_table, mapping_table + global_comm_size);
  auto oob_allgather = oob_factory_(global_rank, global_comm_size, std::move(mapping_table_vec));

  auto ucc_comm = std::make_unique<UCCCommunicator>(
    global_rank, global_comm_size, std::move(oob_allgather), timeout_, lib_.value());

  // Update global_comm with UCC-specific information
  global_comm->unique_id        = unique_id;
  global_comm->status           = true;
  global_comm->global_rank      = global_rank;
  global_comm->global_comm_size = global_comm_size;

  {
    const std::scoped_lock<std::mutex> lock{ucc_comms_lock_};
    ucc_comms_[global_rank] = std::move(ucc_comm);
  }
}

void UCCNetwork::Impl::comm_destroy(legate::comm::coll::CollComm global_comm)
{
  LEGATE_CHECK(lib_.has_value());

  // Look up the communicator by unique ID and extract it safely
  auto ucc_comm_ptr = [&]() -> std::unique_ptr<UCCCommunicator> {
    const std::scoped_lock<std::mutex> lock{ucc_comms_lock_};

    auto it = ucc_comms_.find(global_comm->global_rank);
    if (it == ucc_comms_.end()) {
      LEGATE_ABORT(
        fmt::format("Invalid communicator for comm_destroy, rank: {}", global_comm->global_rank));
    }
    // Move the unique_ptr out of the map to ensure safe access
    auto result = std::move(it->second);
    ucc_comms_.erase(it);
    return result;
  }();

  // Now we can safely destroy the UCC team without holding the lock
  ucc_comm_ptr->destroy_ucc_team_and_context();
  global_comm->status = false;
}

void UCCNetwork::Impl::all_to_all_v(const void* sendbuf,
                                    const int sendcounts[],
                                    const int sdispls[],
                                    void* recvbuf,
                                    const int recvcounts[],
                                    const int rdispls[],
                                    legate::comm::coll::CollDataType type,
                                    legate::comm::coll::CollComm global_comm)
{
  LEGATE_CHECK(lib_.has_value());
  LEGATE_CHECK(global_comm != nullptr);
  LEGATE_CHECK(sendbuf != nullptr);
  LEGATE_CHECK(recvbuf != nullptr);
  LEGATE_CHECK(sendcounts != nullptr);
  LEGATE_CHECK(sdispls != nullptr);
  LEGATE_CHECK(recvcounts != nullptr);
  LEGATE_CHECK(rdispls != nullptr);

  UCCCommunicator* ucc_comm = [&]() -> UCCCommunicator* {
    const std::scoped_lock<std::mutex> lock{ucc_comms_lock_};
    auto it = ucc_comms_.find(global_comm->global_rank);

    if (it == ucc_comms_.end()) {
      LEGATE_ABORT(
        fmt::format("Invalid communicator for all_to_all_v, rank: {}", global_comm->global_rank));
    }

    return it->second.get();
  }();

  const auto ucc_dtype = dtype_to_ucc_dtype_(type);
  // create ucc_count_t arrays for sendcounts, sdispls, recvcounts, rdispls
  // ucc_count_t std::uint64_t
  const auto comm_size = ucc_comm->get_size();
  // using () for the cunstructor to avoid confusion with initializer_list
  std::vector<ucc_count_t> sendcounts_count(sendcounts, sendcounts + comm_size);
  std::vector<ucc_count_t> sdispls_count(sdispls, sdispls + comm_size);
  std::vector<ucc_count_t> recvcounts_count(recvcounts, recvcounts + comm_size);
  std::vector<ucc_count_t> rdispls_count(rdispls, rdispls + comm_size);

  ucc_coll_args_t coll_args{};

  coll_args.mask      = UCC_COLL_ARGS_FIELD_FLAGS;
  coll_args.flags     = UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
  coll_args.coll_type = UCC_COLL_TYPE_ALLTOALLV;
  coll_args.src.info.mem_type        = UCC_MEMORY_TYPE_HOST;
  coll_args.src.info_v.buffer        = const_cast<void*>(sendbuf);
  coll_args.src.info_v.counts        = static_cast<ucc_count_t*>(sendcounts_count.data());
  coll_args.src.info_v.displacements = static_cast<ucc_count_t*>(sdispls_count.data());
  coll_args.src.info_v.datatype      = ucc_dtype;
  coll_args.dst.info_v.buffer        = recvbuf;
  coll_args.dst.info_v.counts        = static_cast<ucc_count_t*>(recvcounts_count.data());
  coll_args.dst.info_v.displacements = static_cast<ucc_count_t*>(rdispls_count.data());
  coll_args.dst.info_v.datatype      = ucc_dtype;

  LEGATE_CHECK_UCC(ucc_comm->ucc_collective(&coll_args));
}

ucc_coll_args_t UCCNetwork::Impl::make_ucc_coll_args_(const void* sendbuf,
                                                      void* recvbuf,
                                                      std::uint64_t send_count,
                                                      std::uint64_t recv_count,
                                                      ucc_coll_type_t coll_type,
                                                      legate::comm::coll::CollDataType type)
{
  ucc_coll_args_t coll_args{};
  const auto ucc_dtype = dtype_to_ucc_dtype_(type);

  coll_args.mask              = UCC_COLL_ARGS_FIELD_FLAGS;
  coll_args.flags             = UCC_COLL_ARGS_FLAG_COUNT_64BIT;
  coll_args.coll_type         = coll_type;
  coll_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
  coll_args.src.info.buffer   = const_cast<void*>(sendbuf);
  coll_args.src.info.count    = send_count;
  coll_args.src.info.datatype = ucc_dtype;
  coll_args.dst.info.buffer   = recvbuf;
  coll_args.dst.info.count    = recv_count;
  coll_args.dst.info.datatype = ucc_dtype;

  return coll_args;
}

void UCCNetwork::Impl::all_to_all(const void* sendbuf,
                                  void* recvbuf,
                                  int count,
                                  legate::comm::coll::CollDataType type,
                                  legate::comm::coll::CollComm global_comm)
{
  LEGATE_CHECK(lib_.has_value());
  LEGATE_CHECK(global_comm != nullptr);
  LEGATE_CHECK(sendbuf != nullptr);
  LEGATE_CHECK(recvbuf != nullptr);

  UCCCommunicator* ucc_comm = [&]() -> UCCCommunicator* {
    const std::scoped_lock<std::mutex> lock{ucc_comms_lock_};
    auto it = ucc_comms_.find(global_comm->global_rank);

    if (it == ucc_comms_.end()) {
      LEGATE_ABORT(
        fmt::format("Invalid communicator for all_to_all_v, rank: {}", global_comm->global_rank));
    }

    return it->second.get();
  }();

  ucc_coll_args_t coll_args =
    make_ucc_coll_args_(sendbuf, recvbuf, count, count, UCC_COLL_TYPE_ALLTOALL, type);
  LEGATE_CHECK_UCC(ucc_comm->ucc_collective(&coll_args));
}

void UCCNetwork::Impl::all_gather(const void* sendbuf,
                                  void* recvbuf,
                                  int count,
                                  legate::comm::coll::CollDataType type,
                                  legate::comm::coll::CollComm global_comm)
{
  LEGATE_CHECK(lib_.has_value());
  LEGATE_CHECK(global_comm != nullptr);
  LEGATE_CHECK(sendbuf != nullptr);
  LEGATE_CHECK(recvbuf != nullptr);

  UCCCommunicator* ucc_comm = [&]() -> UCCCommunicator* {
    const std::scoped_lock<std::mutex> lock{ucc_comms_lock_};
    auto it = ucc_comms_.find(global_comm->global_rank);
    if (it == ucc_comms_.end()) {
      LEGATE_ABORT(
        fmt::format("Invalid communicator for all_gather, rank: {}", global_comm->global_rank));
    }

    return it->second.get();
  }();

  ucc_coll_args_t coll_args =
    make_ucc_coll_args_(sendbuf,
                        recvbuf,
                        count,
                        static_cast<std::uint64_t>(count) * ucc_comm->get_size(),
                        UCC_COLL_TYPE_ALLGATHER,
                        type);

  LEGATE_CHECK_UCC(ucc_comm->ucc_collective(&coll_args));
}

void UCCNetwork::Impl::all_reduce(const void* sendbuf,
                                  void* recvbuf,
                                  int count,
                                  legate::comm::coll::CollDataType type,
                                  ReductionOpKind op,
                                  legate::comm::coll::CollComm global_comm)
{
  LEGATE_CHECK(lib_.has_value());
  LEGATE_CHECK(global_comm != nullptr);
  LEGATE_CHECK(sendbuf != nullptr);
  LEGATE_CHECK(recvbuf != nullptr);
  LEGATE_CHECK(count >= 0);

  UCCCommunicator* ucc_comm = [&]() -> UCCCommunicator* {
    const std::scoped_lock<std::mutex> lock{ucc_comms_lock_};
    const auto it = ucc_comms_.find(global_comm->global_rank);

    if (it == ucc_comms_.end()) {
      LEGATE_ABORT(
        fmt::format("Invalid communicator for all_reduce, rank: {}", global_comm->global_rank));
    }

    return it->second.get();
  }();

  ucc_coll_args_t coll_args =
    make_ucc_coll_args_(sendbuf, recvbuf, count, count, UCC_COLL_TYPE_ALLREDUCE, type);
  coll_args.op = redop_to_ucc_redop_(op);
  LEGATE_CHECK_UCC(ucc_comm->ucc_collective(&coll_args));
}

// UCCNetwork public interface implementation
UCCNetwork::UCCNetwork(OOBAllgatherFactory oob_factory, std::uint32_t timeout)
  : impl_{std::make_unique<Impl>(std::move(oob_factory), timeout)}
{
  LEGATE_CHECK(current_unique_id_ == 0);
  BackendNetwork::coll_inited_ = true;
  BackendNetwork::comm_type    = legate::comm::coll::CollCommType::CollUCC;
}

UCCNetwork::UCCNetwork(std::uint32_t timeout)
  : UCCNetwork{create_mpi_oob_allgather_factory(), timeout}
{
}

UCCNetwork::~UCCNetwork() noexcept
{
  LEGATE_CHECK(BackendNetwork::coll_inited_ == true);
  BackendNetwork::coll_inited_ = false;
}

int UCCNetwork::init_comm()
{
  auto id = get_unique_id_();

  return impl_->init_comm(id);
}

void UCCNetwork::abort() { impl_->abort(); }

void UCCNetwork::comm_create(legate::comm::coll::CollComm global_comm,
                             int global_comm_size,
                             int global_rank,
                             int unique_id,
                             const int* mapping_table)
{
  impl_->comm_create(global_comm, global_comm_size, global_rank, unique_id, mapping_table);
}

void UCCNetwork::comm_destroy(legate::comm::coll::CollComm global_comm)
{
  impl_->comm_destroy(global_comm);
}

void UCCNetwork::all_to_all_v(const void* sendbuf,
                              const int sendcounts[],
                              const int sdispls[],
                              void* recvbuf,
                              const int recvcounts[],
                              const int rdispls[],
                              legate::comm::coll::CollDataType type,
                              legate::comm::coll::CollComm global_comm)
{
  impl_->all_to_all_v(
    sendbuf, sendcounts, sdispls, recvbuf, recvcounts, rdispls, type, global_comm);
}

void UCCNetwork::all_to_all(const void* sendbuf,
                            void* recvbuf,
                            int count,
                            legate::comm::coll::CollDataType type,
                            legate::comm::coll::CollComm global_comm)
{
  impl_->all_to_all(sendbuf, recvbuf, count, type, global_comm);
}

void UCCNetwork::all_gather(const void* sendbuf,
                            void* recvbuf,
                            int count,
                            legate::comm::coll::CollDataType type,
                            legate::comm::coll::CollComm global_comm)
{
  impl_->all_gather(sendbuf, recvbuf, count, type, global_comm);
}

void UCCNetwork::all_reduce(const void* sendbuf,
                            void* recvbuf,
                            int count,
                            legate::comm::coll::CollDataType type,
                            ReductionOpKind op,
                            legate::comm::coll::CollComm global_comm)
{
  impl_->all_reduce(sendbuf, recvbuf, count, type, op, global_comm);
}

void UCCNetwork::shutdown() { impl_->shutdown(); }

UCCCommunicator::UCCCommunicator(int global_rank,
                                 std::size_t global_size,
                                 std::unique_ptr<OOBAllgather> allgather,
                                 std::uint32_t timeout,
                                 ucc_lib_h lib)
  : size_{global_size},
    context_{create_ucc_ctx_(global_rank, global_size, lib, allgather.get())},
    team_{create_ucc_team_(global_rank, global_size, context_, allgather.get(), timeout)},
    oob_allgather_{std::move(allgather)},
    timeout_{timeout}
{
}

void UCCNetwork::Impl::init_ucc_lib_()
{
  ucc_lib_config_h lib_config;
  ucc_lib_params_t lib_params{};
  ucc_status_t status;

  // we don't use env_prefix and config_file, so pass nullptr
  LEGATE_CHECK_UCC(
    ucc_lib_config_read(/*env_prefix*/ nullptr, /*config_file*/ nullptr, &lib_config));

  lib_params.mask            = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode     = UCC_THREAD_MULTIPLE;
  lib_params.coll_types      = {};
  lib_params.reduction_types = {};
  lib_params.sync_type       = {};

  ucc_lib_h lib_handle;

  status = ucc_init(&lib_params, lib_config, &lib_handle);
  lib_   = lib_handle;
  ucc_lib_config_release(lib_config);

  if (status != UCC_OK) {
    LEGATE_ABORT(fmt::format("Failed to initialize UCC library: {}", ucc_status_string(status)));
  }

  logger().debug() << "UCC library initialized successfully";
}

ucc_context_h UCCCommunicator::create_ucc_ctx_(int rank,
                                               std::size_t size,
                                               ucc_lib_h lib,
                                               OOBAllgather* oob_allgather)
{
  ucc_context_config_h ctx_config;

  // we don't use config_file, so pass nullptr
  LEGATE_CHECK_UCC(ucc_context_config_read(lib, /*config_file*/ nullptr, &ctx_config));

  ucc_context_params_t ctx_params{};

  ctx_params.mask          = UCC_CONTEXT_PARAM_FIELD_OOB;
  ctx_params.type          = UCC_CONTEXT_SHARED;
  ctx_params.oob.allgather = OOBAllgather::oob_allgather;
  ctx_params.oob.req_test  = OOBAllgather::oob_test;
  ctx_params.oob.req_free  = OOBAllgather::oob_free;
  ctx_params.oob.coll_info = oob_allgather;
  ctx_params.oob.n_oob_eps = static_cast<std::uint32_t>(size);
  ctx_params.oob.oob_ep    = static_cast<std::uint32_t>(rank);

  ucc_context_h ctx;

  const ucc_status_t status = ucc_context_create(lib, &ctx_params, ctx_config, &ctx);
  ucc_context_config_release(ctx_config);

  if (status != UCC_OK) {
    LEGATE_ABORT(
      fmt::format("Failed to create UCC context for communicator: {}", ucc_status_string(status)));
  }

  return ctx;
}

ucc_team_h UCCCommunicator::create_ucc_team_(
  int rank, std::size_t size, ucc_context_h ctx, OOBAllgather* oob_allgather, std::uint32_t timeout)
{
  ucc_team_params_t team_params{};

  team_params.mask          = UCC_TEAM_PARAM_FIELD_OOB;
  team_params.ordering      = UCC_COLLECTIVE_INIT_AND_POST_UNORDERED;
  team_params.team_size     = size;
  team_params.sync_type     = UCC_NO_SYNC_COLLECTIVES;
  team_params.oob.allgather = OOBAllgather::oob_allgather;
  team_params.oob.req_test  = OOBAllgather::oob_test;
  team_params.oob.req_free  = OOBAllgather::oob_free;
  team_params.oob.coll_info = oob_allgather;
  team_params.oob.n_oob_eps = static_cast<std::int32_t>(size);
  team_params.oob.oob_ep    = static_cast<std::int32_t>(rank);

  ucc_team_h team;

  LEGATE_CHECK_UCC(ucc_team_create_post(&ctx, 1, &team_params, &team));

  // Wait for team creation to complete
  const auto start_time  = std::chrono::steady_clock::now();
  const auto timeout_sec = std::chrono::seconds{timeout};
  ucc_status_t status;

  while ((status = ucc_team_create_test(team)) == UCC_INPROGRESS) {
    if (std::chrono::steady_clock::now() - start_time > timeout_sec) {
      LEGATE_ABORT(fmt::format("UCC team creation timed out after {} seconds.", timeout));
    }
    ucc_context_progress(ctx);
  }

  if (status != UCC_OK) {
    LEGATE_ABORT(fmt::format("Failed to create UCC team. Error: {}", ucc_status_string(status)));
  }
  logger().debug() << fmt::format("UCC team created successfully (rank={}, size={})", rank, size);

  return team;
}

void UCCCommunicator::destroy_ucc_team_and_context()
{
  ucc_status_t st;
  const auto start_time = std::chrono::steady_clock::now();
  const auto timeout    = std::chrono::seconds{timeout_};

  // try until timeout
  do {
    st = ucc_team_destroy(team_);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      logger().debug() << fmt::format("UCC team destruction timed out after {} seconds.", timeout_);
    }
  } while (st == UCC_INPROGRESS);

  st = ucc_context_destroy(context_);
  if (st != UCC_OK) {
    logger().debug() << fmt::format("Failed to destroy UCC context: {}", ucc_status_string(st));
  }
}

ucc_datatype_t UCCNetwork::Impl::dtype_to_ucc_dtype_(legate::comm::coll::CollDataType dtype)
{
  switch (dtype) {
    case legate::comm::coll::CollDataType::CollInt8: [[fallthrough]];
    case legate::comm::coll::CollDataType::CollChar: return UCC_DT_INT8;
    case legate::comm::coll::CollDataType::CollUint8: return UCC_DT_UINT8;
    case legate::comm::coll::CollDataType::CollInt: return UCC_DT_INT32;
    case legate::comm::coll::CollDataType::CollUint32: return UCC_DT_UINT32;
    case legate::comm::coll::CollDataType::CollInt64: return UCC_DT_INT64;
    case legate::comm::coll::CollDataType::CollUint64: return UCC_DT_UINT64;
    case legate::comm::coll::CollDataType::CollFloat: return UCC_DT_FLOAT32;
    case legate::comm::coll::CollDataType::CollDouble: return UCC_DT_FLOAT64;
  }
  LEGATE_ABORT("Unknown datatype");
}

ucc_reduction_op_t UCCNetwork::Impl::redop_to_ucc_redop_(ReductionOpKind op)
{
  switch (op) {
    case legate::ReductionOpKind::ADD: return UCC_OP_SUM;
    case legate::ReductionOpKind::MUL: return UCC_OP_PROD;
    case legate::ReductionOpKind::MAX: return UCC_OP_MAX;
    case legate::ReductionOpKind::MIN: return UCC_OP_MIN;
    case legate::ReductionOpKind::AND: return UCC_OP_BAND;
    case legate::ReductionOpKind::OR: return UCC_OP_BOR;
    case legate::ReductionOpKind::XOR: return UCC_OP_BXOR;
  }
  LEGATE_ABORT(fmt::format("Unknown reduction operation: {}", static_cast<std::int32_t>(op)));
}

std::size_t UCCNetwork::Impl::get_dtype_size_(legate::comm::coll::CollDataType dtype)
{
  switch (dtype) {
    case legate::comm::coll::CollDataType::CollInt8: [[fallthrough]];
    case legate::comm::coll::CollDataType::CollChar: [[fallthrough]];
    case legate::comm::coll::CollDataType::CollUint8: return sizeof(std::uint8_t);
    case legate::comm::coll::CollDataType::CollInt: return sizeof(int);
    case legate::comm::coll::CollDataType::CollUint32: return sizeof(std::uint32_t);
    case legate::comm::coll::CollDataType::CollInt64: return sizeof(std::int64_t);
    case legate::comm::coll::CollDataType::CollUint64: return sizeof(std::uint64_t);
    case legate::comm::coll::CollDataType::CollFloat: return sizeof(float);
    case legate::comm::coll::CollDataType::CollDouble: return sizeof(double);
  }
  LEGATE_ABORT("Unknown datatype");
}

ucc_status_t UCCCommunicator::ucc_collective(ucc_coll_args_t* coll_args)
{
  ucc_coll_req_h req;

  auto status = ucc_collective_init(coll_args, &req, team_);

  if (status != UCC_OK) {
    return status;
  }

  status = ucc_collective_post(req);
  if (status != UCC_OK) {
    ucc_collective_finalize(req);
    return status;
  }

  status = ucc_collective_test(req);
  while (status == UCC_INPROGRESS) {
    status = ucc_context_progress(context_);
    if (status != UCC_OK && status != UCC_INPROGRESS) {
      ucc_collective_finalize(req);
      return status;
    }
    status = ucc_collective_test(req);
  }

  ucc_collective_finalize(req);
  return status;
}

std::size_t UCCCommunicator::get_size() const { return size_; }

}  // namespace legate::detail::comm::coll
