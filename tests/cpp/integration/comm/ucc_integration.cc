/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/coll.h>
#include <legate/comm/coll_comm.h>
#include <legate/comm/detail/mpi_interface.h>
#include <legate/comm/detail/mpi_oob_allgather.h>
#include <legate/comm/detail/oob_allgather.h>
#include <legate/comm/detail/ucc_network.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utilities/utilities.h>
#include <vector>

namespace ucc_integration_test {

using legate::detail::comm::coll::BackendNetwork;
using legate::detail::comm::coll::OOBAllgather;
using legate::detail::comm::coll::UCCNetwork;

/**
 * @brief This class will be shared by all ranks and will be used to allgather data.
 * This is used in a non-mpi environment.
 */
class SharedMemoryAllgather {
 public:
  explicit SharedMemoryAllgather(std::size_t size) : expected_count_{size} {}

  /**
   * @brief This is a custom allgather that works through memory copy for testing.
   * This is used in a non-mpi environment.
   */
  bool allgather(const void* sendbuf, std::size_t message_size, void* recvbuf, int rank, int round)
  {
    std::unique_lock<std::mutex> lock{mtx_};
    // buffer is kept for each round and rank <round, <rank, buffer>>
    buffers_[round][rank] = sendbuf;

    if (buffers_[round].size() == expected_count_) {
      // Last thread: prepare result for ALL threads
      cv_.notify_all();
    }

    // All threads wait here until the result is ready
    cv_.wait(lock, [this, round] { return buffers_[round].size() == expected_count_; });

    // All threads copy the complete result to their own recvbuf
    for (std::size_t i = 0; i < expected_count_; i++) {
      std::memcpy(static_cast<std::uint8_t*>(recvbuf) + (i * message_size),
                  buffers_[round][static_cast<int>(i)],
                  message_size);
    }
    if (round > 1) {
      buffers_.erase(round - 1);
    }
    return true;
  }

 private:
  // send buffers for each and round
  std::unordered_map<int, std::unordered_map<int, const void*>> buffers_{};
  mutable std::mutex mtx_{};
  mutable std::condition_variable cv_{};
  std::size_t expected_count_{};
};

/**
 * @brief This is a custom OOBAllgather that works through memory copy for testing.
 * This is used in a non-mpi environment.
 */
class MemoryCopyOOBAllgather final : public OOBAllgather {
 public:
  MemoryCopyOOBAllgather(int rank, std::shared_ptr<SharedMemoryAllgather> shared_memory_allgather)
    : rank_{rank}, shared_memory_allgather_{std::move(shared_memory_allgather)}
  {
  }

  ucc_status_t allgather(const void* sendbuf,
                         void* recvbuf,
                         std::size_t message_size,
                         void* /*allgather_info*/,
                         void** /*request*/) override
  {
    shared_memory_allgather_->allgather(sendbuf, message_size, recvbuf, rank_, allgather_count_++);
    return UCC_OK;
  }

  ucc_status_t test(void* /*request*/) override { return UCC_OK; }

  ucc_status_t free(void* /*request*/) override { return UCC_OK; }

 private:
  // rank within the shared memory allgather
  int rank_{};
  // number of threads
  std::shared_ptr<SharedMemoryAllgather> shared_memory_allgather_{};
  int allgather_count_{0};
};

// Integration test fixture
class UCCIntegrationTest : public DefaultFixture {
 protected:
  void SetUp() override
  {
    const int num_threads = 4;

    DefaultFixture::SetUp();
    initialize_mpi_();
    setup_ucc_(num_threads);
  }

  void setup_ucc_(int num_threads)
  {
    // if mpi is not initialized, use shared memory allgather
    if (!mpi_initialized_) {
      shared_memory_allgather_ = std::make_shared<SharedMemoryAllgather>(num_threads);
      BackendNetwork::create_network(std::make_unique<legate::detail::comm::coll::UCCNetwork>(
        [this](int global_rank,
               int /*global_size*/,
               const std::vector<int>& /*table*/) -> std::unique_ptr<OOBAllgather> {
          return std::make_unique<MemoryCopyOOBAllgather>(global_rank, shared_memory_allgather_);
        }));
    } else {
      // create the ucc network with mpi oob allgather
      BackendNetwork::create_network(std::make_unique<legate::detail::comm::coll::UCCNetwork>());
    }

    num_threads_      = num_threads;
    global_size_      = mpi_size_ * num_threads;
    local_rank_start_ = mpi_rank_ * num_threads;

    const auto unique_id = 10245;

    mapping_table_.resize(global_size_);
    std::generate(mapping_table_.begin(), mapping_table_.end(), [i = 0, num_threads]() mutable {
      return i++ / num_threads;
    });

    ucc_network_ =
      dynamic_cast<legate::detail::comm::coll::UCCNetwork*>(BackendNetwork::get_network().get());

    ASSERT_TRUE(ucc_network_ != nullptr);
    ASSERT_EQ(ucc_network_->init_comm(), 0);

    // create a communicator for each thread
    std::mutex comm_mutex;
    std::vector<std::thread> threads;

    for (int i = local_rank_start_; i < local_rank_start_ + num_threads_; i++) {
      std::thread t{[this, i, unique_id, &comm_mutex]() {
        try {
          auto comm = std::make_unique<legate::comm::coll::Coll_Comm>();

          ucc_network_->comm_create(comm.get(), global_size_, i, unique_id, mapping_table_.data());
          ASSERT_EQ(comm->unique_id, unique_id);
          ASSERT_EQ(comm->status, true);

          const std::scoped_lock<std::mutex> lock{comm_mutex};

          comms_[i] = std::move(comm);
        } catch (const std::exception& e) {
          FAIL() << "Error creating comm for rank " << i << ": " << e.what();
        }
      }};

      threads.push_back(std::move(t));
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  /**
   * @brief Check if MPI is initialized and get rank and size.
   */
  void initialize_mpi_()
  {
    // initialize mpi
    int init_flag    = 0;
    auto status      = legate::detail::comm::mpi::detail::MPIInterface::mpi_initialized(&init_flag);
    mpi_initialized_ = init_flag;
    if (!mpi_initialized_) {
      mpi_rank_ = 0;
      mpi_size_ = 1;
      // if MPI is not initialized, we will use the shared memory allgather
      return;
    }

    ASSERT_EQ(status, legate::detail::comm::mpi::detail::MPIInterface::MPI_SUCCESS());
    status = legate::detail::comm::mpi::detail::MPIInterface::mpi_comm_rank(
      legate::detail::comm::mpi::detail::MPIInterface::MPI_COMM_WORLD(), &mpi_rank_);
    ASSERT_EQ(status, legate::detail::comm::mpi::detail::MPIInterface::MPI_SUCCESS());
    status = legate::detail::comm::mpi::detail::MPIInterface::mpi_comm_size(
      legate::detail::comm::mpi::detail::MPIInterface::MPI_COMM_WORLD(), &mpi_size_);
    ASSERT_EQ(status, legate::detail::comm::mpi::detail::MPIInterface::MPI_SUCCESS());
  }

  int mpi_rank_{}, mpi_size_{};
  bool mpi_initialized_ = false;
  int global_size_{};
  int num_threads_{};
  int local_rank_start_{};

  std::unordered_map<int, std::unique_ptr<legate::comm::coll::Coll_Comm>> comms_{};
  std::vector<int> mapping_table_{};
  legate::detail::comm::coll::UCCNetwork* ucc_network_{};
  std::shared_ptr<SharedMemoryAllgather> shared_memory_allgather_{};
};

// Type-to-CollDataType mapping
template <typename T>
struct TypeToCollDataType;

template <>
struct TypeToCollDataType<std::int8_t> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollInt8;
};

template <>
struct TypeToCollDataType<std::int32_t> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollInt;
};

template <>
struct TypeToCollDataType<std::int64_t> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollInt64;
};

template <>
struct TypeToCollDataType<double> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollDouble;
};

template <>
struct TypeToCollDataType<float> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollFloat;
};

// Define the types we want to test
using TestTypes = ::testing::Types<std::int8_t, std::int32_t, std::int64_t, double, float>;

// Typed test fixture for allgather, alltoall, and alltoallv
template <typename T>
class UCCIntegrationTypedTest : public UCCIntegrationTest {
 protected:
  static constexpr auto COLL_TYPE = TypeToCollDataType<T>::VALUE;

  /**
   * @brief Performs all-gather operation with verification for a specific rank
   *
   * Test Pattern:
   * - Each rank creates a send buffer with 4 elements containing sequential values
   *   starting from (rank * 4). For example:
   *   - Rank 0 sends: [0, 1, 2, 3]
   *   - Rank 1 sends: [4, 5, 6, 7]
   *   - Rank 2 sends: [8, 9, 10, 11]
   * - After all-gather, each rank should have a receive buffer containing
   *   concatenated data from all ranks in rank order: [0,1,2,3,4,5,6,7,8,9,10,11,...]
   *
   * @param global_rank The rank performing the operation
   */
  void perform_allgather_test_(int global_rank)
  {
    constexpr std::int32_t num_elements = 4;
    std::vector<T> send_data(num_elements);

    T start_value = global_rank;

    start_value *= num_elements;
    std::iota(send_data.begin(), send_data.end(), start_value);

    std::vector<T> recv_data(num_elements * this->global_size_);

    this->ucc_network_->all_gather(
      send_data.data(), recv_data.data(), num_elements, COLL_TYPE, this->comms_[global_rank].get());

    constexpr double tolerance = 1e-10;

    for (int i = 0; i < this->global_size_ * num_elements; i++) {
      if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        EXPECT_NEAR(recv_data[i], static_cast<T>(i), tolerance);
      } else {
        EXPECT_EQ(recv_data[i], static_cast<T>(i));
      }
    }
  }

  /**
   * @brief Performs all-to-all operation with verification for a specific rank
   *
   * Test Pattern:
   * - Each rank sends the same data (its rank value) to all other ranks
   * - Send buffer size: global_size * 8 elements, all filled with sender's rank value
   * - For example, with 3 ranks:
   *   - Rank 0 sends: [0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0] (24 zeros)
   *   - Rank 1 sends: [1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1] (24 ones)
   *   - Rank 2 sends: [2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2] (24 twos)
   * - After all-to-all, each rank receives 8 elements from each sender:
   *   - All ranks receive: [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2]
   *
   * @param global_rank The rank performing the operation
   */
  void perform_alltoall_test_(int global_rank)
  {
    constexpr std::int32_t num_elements_per_rank = 8;
    const std::int32_t num_elements              = this->global_size_ * num_elements_per_rank;
    std::vector<T> send_data(num_elements);

    std::fill(send_data.begin(), send_data.end(), static_cast<T>(global_rank));

    std::vector<T> recv_data(num_elements);

    this->ucc_network_->all_to_all(
      send_data.data(), recv_data.data(), num_elements, COLL_TYPE, this->comms_[global_rank].get());

    constexpr double tolerance = 1e-10;

    // Verify that data from each rank is received correctly
    // Each rank has value start from rank...rank+num_elements_per_rank-1
    for (int source_rank = 0; source_rank < this->global_size_; source_rank++) {
      for (int elem = 0; elem < num_elements_per_rank; elem++) {
        const int index = (source_rank * num_elements_per_rank) + elem;
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
          EXPECT_NEAR(recv_data[index], static_cast<T>(source_rank), tolerance);
        } else {
          EXPECT_EQ(recv_data[index], static_cast<T>(source_rank));
        }
      }
    }
  }

  /**
   * @brief Performs all-to-all-v (variable) operation with verification for a specific rank
   *
   * Test Pattern:
   * - Each rank sends different amounts of data to each destination rank
   * - Send counts: rank R sends (R+1) elements to each other rank
   * - Receive counts: rank R receives (S+1) elements from each sender rank S
   * - Data pattern: sender rank S fills its data with (S * 1000 + local_index)
   *
   * Example with 3 ranks:
   * - Rank 0 sends 1 element to each rank: [0] to rank 0, [1] to rank 1, [2] to rank 2
   * - Rank 1 sends 2 elements to each rank: [1000,1001] to rank 0, [1002,1003] to rank 1,
   * [1004,1005] to rank 2
   * - Rank 2 sends 3 elements to each rank: [2000,2001,2002] to rank 0, [2003,2004,2005] to rank 1,
   * [2006,2007,2008] to rank 2
   *
   * After all-to-all-v, each rank receives:
   * - Rank 0 receives: [0, 1000,1001, 2000,2001,2002] (1+2+3=6 elements total)
   * - Rank 1 receives: [1, 1002,1003, 2003,2004,2005] (1+2+3=6 elements total)
   * - Rank 2 receives: [2, 1004,1005, 2006,2007,2008] (1+2+3=6 elements total)
   *
   * @param global_rank The rank performing the operation
   */
  void perform_alltoallv_test_(int global_rank)
  {
    // Each rank sends different amounts to each other rank
    std::vector<int> sendcounts(this->global_size_);
    std::vector<int> sdispls(this->global_size_);
    std::vector<int> recvcounts(this->global_size_);
    std::vector<int> rdispls(this->global_size_);

    // Setup send counts: rank i sends (i+1) elements to rank j
    int total_send = 0;

    for (int i = 0; i < this->global_size_; ++i) {
      sendcounts[i] = global_rank + 1;  // Send rank+1 elements to each process
      sdispls[i]    = total_send;
      total_send += sendcounts[i];
    }

    // Setup recv counts: rank i receives (j+1) elements from rank j
    int total_recv = 0;

    for (int i = 0; i < this->global_size_; ++i) {
      recvcounts[i] = i + 1;  // Receive i+1 elements from rank i
      rdispls[i]    = total_recv;
      total_recv += recvcounts[i];
    }

    // Prepare send buffer: each destination gets data pattern (sender_rank * 1000 + local_index)
    const int pattern_offset = 1000;
    std::vector<T> sendbuf(total_send);

    for (int dest = 0; dest < this->global_size_; ++dest) {
      const int dest_start = sdispls[dest];
      const int dest_count = sendcounts[dest];

      for (int i = 0; i < dest_count; ++i) {
        sendbuf[static_cast<std::size_t>(dest_start) + i] =
          static_cast<T>(static_cast<std::size_t>((global_rank * pattern_offset) + i));
      }
    }

    // Prepare receive buffer
    std::vector<T> recvbuf(total_recv);

    this->ucc_network_->all_to_all_v(sendbuf.data(),
                                     sendcounts.data(),
                                     sdispls.data(),
                                     recvbuf.data(),
                                     recvcounts.data(),
                                     rdispls.data(),
                                     COLL_TYPE,
                                     this->comms_[global_rank].get());

    constexpr double tolerance = 1e-10;
    // Verify received data
    for (int sender = 0; sender < this->global_size_; ++sender) {
      const int start_idx = rdispls[sender];
      const int count     = recvcounts[sender];

      for (int i = 0; i < count; ++i) {
        const auto expected = static_cast<T>(static_cast<std::size_t>(sender * pattern_offset) + i);
        const auto received = recvbuf[static_cast<std::size_t>(start_idx) + i];
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
          EXPECT_NEAR(received, expected, tolerance);
        } else {
          EXPECT_EQ(received, expected);
        }
      }
    }
  }
};

TYPED_TEST_SUITE(UCCIntegrationTypedTest, TestTypes, ::testing::internal::DefaultNameGenerator);

TYPED_TEST(UCCIntegrationTypedTest, AllGather)
{
  std::vector<std::thread> threads;
  // create n threads
  for (int global_rank = this->local_rank_start_;
       global_rank < this->local_rank_start_ + this->num_threads_;
       global_rank++) {
    std::thread t{[this, global_rank]() { this->perform_allgather_test_(global_rank); }};
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }
}

TYPED_TEST(UCCIntegrationTypedTest, AllToAll)
{
  std::vector<std::thread> threads;
  // create threads for each rank for alltoall
  for (int global_rank = this->local_rank_start_;
       global_rank < this->local_rank_start_ + this->num_threads_;
       global_rank++) {
    std::thread t{[this, global_rank]() { this->perform_alltoall_test_(global_rank); }};
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }
}

TYPED_TEST(UCCIntegrationTypedTest, AllToAllV)
{
  std::vector<std::thread> threads;
  // create n threads
  for (int global_rank = this->local_rank_start_;
       global_rank < this->local_rank_start_ + this->num_threads_;
       global_rank++) {
    std::thread t{[this, global_rank]() { this->perform_alltoallv_test_(global_rank); }};
    threads.push_back(std::move(t));
  }

  for (auto& t : threads) {
    t.join();
  }
}

}  // namespace ucc_integration_test
