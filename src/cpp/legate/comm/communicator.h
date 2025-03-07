/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <legion.h>

/**
 * @file
 * @brief Class definition for legate::comm::Communicator
 */

namespace legate::comm {

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief A thin wrapper class for communicators stored in futures. This class only provides
 * a template method to retrieve the communicator handle and the client is expected to pass
 * the right handle type.
 *
 * The following is the list of handle types for communicators supported in Legate:
 *
 *   - NCCL: ncclComm_t*
 *   - CPU communicator in Legate: legate::comm::coll::CollComm*
 *   - CAL: cal_comm_t
 */
class Communicator {
 public:
  Communicator() = default;
  explicit Communicator(Legion::Future future);

  Communicator(const Communicator&)            = default;
  Communicator& operator=(const Communicator&) = default;

  /**
   * @brief Returns the communicator stored in the wrapper
   *
   * @tparam T The type of communicator handle to get (see valid types above)
   *
   * @return A communicator
   */
  template <typename T>
  [[nodiscard]] T get() const;

 private:
  Legion::Future future_{};
};

/** @} */

}  // namespace legate::comm

#include <legate/comm/communicator.inl>
