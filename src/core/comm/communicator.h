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

#include "legion.h"

/**
 * @file
 * @brief Class definition for legate::comm::Communicator
 */

namespace legate::comm {

/**
 * @ingroup task
 * @brief A thin wrapper class for communicators stored in futures. This class only provides
 * a tempalte method to retrieve the communicator handle and the client is expected to pass
 * the right handle type.
 *
 * The following is the list of handle types for communicators supported in Legate:
 *
 *   - NCCL: ncclComm_t*
 *   - CPU communicator in Legate: legate::comm::coll::CollComm*
 */
class Communicator {
 public:
  Communicator() {}
  Communicator(Legion::Future future) : future_(future) {}

 public:
  Communicator(const Communicator&)            = default;
  Communicator& operator=(const Communicator&) = default;

 public:
  /**
   * @brief Returns the communicator stored in the wrapper
   *
   * @tparam T The type of communicator handle to get (see valid types above)
   *
   * @return A communicator
   */
  template <typename T>
  T get() const
  {
    return future_.get_result<T>();
  }

 private:
  Legion::Future future_{};
};

}  // namespace legate::comm
