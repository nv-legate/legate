/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/ucc_network.h>

#include <legate/comm/detail/oob_allgather.h>
#include <legate/utilities/detail/traced_exception.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <utilities/utilities.h>
#include <vector>

namespace ucc_network_test {

class UCCNetworkTest : public DefaultFixture {};

using legate::detail::comm::coll::OOBAllgather;
using legate::detail::comm::coll::UCCNetwork;

/**
 * @brief This class will be shared by all ranks and will be used to allgather data.
 * This is used in a non-mpi environment.
 */
class SharedMemoryAllgather {
  // send buffers kept for each round and rank <round, <rank, buffer>>
  std::unordered_map<int, std::unordered_map<int, const void*>> buffers_{};
  std::mutex mtx_{};
  std::condition_variable cv_{};
  std::size_t expected_count_{};
  std::atomic<int> allgather_call_count_{0};

 public:
  explicit SharedMemoryAllgather(std::size_t size) : expected_count_{size} {}

  /**
   * @brief This is a custom allgather that works through memory copy for testing.
   * This is used in a non-mpi environment.
   */
  void allgather(const void* sendbuf, std::size_t message_size, void* recvbuf, int rank, int round)
  {
    std::unique_lock<std::mutex> lock{mtx_};
    // Add our data to the buffers_ map for the given round and rank
    buffers_[round][rank] = sendbuf;
    if (buffers_[round].size() == expected_count_) {
      // Last thread: notify all other threads that the result is ready
      cv_.notify_all();
    }

    // All threads wait here until result is ready
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
    allgather_call_count_++;
  }

  int get_allgather_call_count() const { return allgather_call_count_.load(); }
};

/**
 * @brief This is a custom OOBAllgather that works through memory copy for testing.
 * This is used in a non-mpi environment.
 */
class MockOOBForUCC final : public OOBAllgather {
  // rank within the shared memory allgather
  int rank_{};
  // number of threads
  std::vector<int> mapping_table_{};
  std::shared_ptr<SharedMemoryAllgather> shared_memory_allgather_;
  int allgather_count_{0};
  bool should_fail_allgather_{false};

 public:
  MockOOBForUCC(int rank,
                int /*size*/,
                const std::vector<int>& mapping_table,
                std::shared_ptr<SharedMemoryAllgather> shared_memory_allgather,
                bool should_fail_allgather = false)
    : rank_{rank},
      mapping_table_(mapping_table),
      shared_memory_allgather_{std::move(shared_memory_allgather)},
      should_fail_allgather_{should_fail_allgather}
  {
  }

  ucc_status_t allgather(const void* sendbuf,
                         void* recvbuf,
                         std::size_t message_size,
                         void* /*allgather_info*/,
                         void** /*request*/) override
  {
    if (should_fail_allgather_) {
      return UCC_ERR_NO_MESSAGE;
    }
    shared_memory_allgather_->allgather(sendbuf, message_size, recvbuf, rank_, allgather_count_);
    allgather_count_++;
    return UCC_OK;
  }

  ucc_status_t test(void* /*request*/) override { return UCC_OK; }

  ucc_status_t free(void* /*request*/) override { return UCC_OK; }
};

// Test UCCNetwork construction and basic properties
TEST_F(UCCNetworkTest, ConstructionAndBasicProperties)
{
  UCCNetwork network;

  EXPECT_EQ(network.init_comm(), 0);
}

TEST_F(UCCNetworkTest, OOBOneThread)
{
  const std::shared_ptr<SharedMemoryAllgather> shared_memory_allgather =
    std::make_shared<SharedMemoryAllgather>(1);
  UCCNetwork network{
    [shared_memory_allgather](int global_rank,
                              int global_size,
                              const std::vector<int>& table) -> std::unique_ptr<OOBAllgather> {
      return std::make_unique<MockOOBForUCC>(
        global_rank, global_size, table, shared_memory_allgather);
    }};

  EXPECT_EQ(network.init_comm(), 0);

  auto comm = std::make_unique<legate::comm::coll::Coll_Comm>();
  std::vector<int> mapping_table(1, 0);
  const int unique_id = 10245;

  network.comm_create(
    comm.get(), /*global_comm_size=*/1, /*global_rank=*/0, unique_id, mapping_table.data());
  EXPECT_EQ(comm->unique_id, unique_id);
  EXPECT_EQ(comm->status, true);
  // No allgather calls should have been made as only one context
  EXPECT_EQ(shared_memory_allgather->get_allgather_call_count(), 0);
}

// Test OOB allgather failure scenarios
TEST_F(UCCNetworkTest, OOBTwoThreads)
{
  const int num_threads = 2;
  const std::shared_ptr<SharedMemoryAllgather> shared_memory_allgather =
    std::make_shared<SharedMemoryAllgather>(num_threads);
  UCCNetwork network{
    [shared_memory_allgather](int global_rank,
                              int global_size,
                              const std::vector<int>& table) -> std::unique_ptr<OOBAllgather> {
      return std::make_unique<MockOOBForUCC>(
        global_rank, global_size, table, shared_memory_allgather, false);
    }};

  EXPECT_EQ(network.init_comm(), 0);

  std::vector<std::unique_ptr<legate::comm::coll::Coll_Comm>> comms;
  std::vector<std::thread> threads;
  std::mutex comm_mutex;
  const int unique_id = 4321;

  for (int i = 0; i < num_threads; i++) {
    std::thread t{[&, index = i]() {
      auto comm = std::make_unique<legate::comm::coll::Coll_Comm>();
      std::vector<int> mapping_table(num_threads, 0);

      network.comm_create(comm.get(), num_threads, index, unique_id, mapping_table.data());

      const std::scoped_lock<std::mutex> lock{comm_mutex};

      comms.push_back(std::move(comm));
    }};
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) {
    t.join();
  }

  std::vector<std::thread> destroy_threads;

  for (int i = 0; i < num_threads; i++) {
    std::thread t{[&, index = i]() {
      EXPECT_EQ(comms[index]->unique_id, unique_id);
      EXPECT_EQ(comms[index]->status, true);
      network.comm_destroy(comms[index].get());
    }};
    destroy_threads.push_back(std::move(t));
  }

  EXPECT_EQ(shared_memory_allgather->get_allgather_call_count(), 6);
  for (auto& t : destroy_threads) {
    t.join();
  }
}

}  // namespace ucc_network_test
