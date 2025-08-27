/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/memory.h>

#include <ucc/api/ucc.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace legate::detail::comm::coll {

/**
 * @brief Out-of-band allgather interface
 *
 * This class provides allgather functionality that can be used for
 * UCC team bootstrap before the UCC collective communication is available. It provides
 * the three methods: allgather, test, and free that are required by UCC.
 */
class OOBAllgather {
 public:
  /**
   * @brief Default constructor
   */
  OOBAllgather() = default;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes
   */
  virtual ~OOBAllgather() = default;

  OOBAllgather(const OOBAllgather&)                = delete;
  OOBAllgather& operator=(const OOBAllgather&)     = delete;
  OOBAllgather(OOBAllgather&&) noexcept            = delete;
  OOBAllgather& operator=(OOBAllgather&&) noexcept = delete;

  /**
   * @brief This is provided to the UCC allgather interface. It internally calls the
   * allgather method of the implementation. This is a static method so that it can be passed
   * into the UCC library.
   *
   * @param send_buf Send buffer
   * @param recv_buf Receive buffer
   * @param size Size in bytes of the message to send
   * @param allgather_info Allgather info. This is a pointer to the OOBAllgather instance.
   * @param request Request pointer, this need to be set, if the allgather is non-blocking so that
   * test and free can be called.
   *
   * @return UCC status, UCC_OK if successful
   */
  [[nodiscard]] static ucc_status_t oob_allgather(void* send_buf,
                                                  void* recv_buf,
                                                  std::size_t size,
                                                  void* allgather_info,
                                                  void** request) noexcept;

  /**
   * @brief This is provided to the UCC free interface. It internally calls the
   * free method of the implementation. This is a static method so that it can be passed
   * into the UCC library.
   *
   * @param request Request pointer created by the allgather method
   *
   * @return UCC status, UCC_OK if successful
   */
  [[nodiscard]] static ucc_status_t oob_free(void* request) noexcept;

  /**
   * @brief This is provided to the UCC test interface. It internally calls the
   * test method of the implementation. This is a static method so that it can be passed
   * into the UCC library.
   *
   * @param request Request pointer created by the allgather method
   *
   * @return UCC status, UCC_OK if successful
   */
  [[nodiscard]] static ucc_status_t oob_test(void* request) noexcept;

  /**
   * @brief Perform out-of-band allgather operation
   *
   * @param sendbuf Input buffer containing data to be gathered from this rank
   * @param recvbuf Output buffer to receive gathered data from all ranks
   * @param message_size Size in bytes of the message to send
   * @param allgather_info Allgather info. This is a pointer to the OOBAllgather instance.
   * @param request Request pointer, this need to be set, if the allgather is non-blocking so that
   * test and free can be called.
   *
   * @return UCC status, UCC_OK if successful
   */
  [[nodiscard]] virtual ucc_status_t allgather(const void* sendbuf,
                                               void* recvbuf,
                                               std::size_t message_size,
                                               void* allgather_info,
                                               void** request) = 0;

  /**
   * @brief Test if the request is completed
   *
   * @param request Pointer to the request object

   * @return UCC status, UCC_OK if successful
   */
  [[nodiscard]] virtual ucc_status_t test(void* request) = 0;

  /**
   * @brief Free the request object
   *
   * @param request Pointer to the request object

   * @return UCC status, UCC_OK if successful
   */
  [[nodiscard]] virtual ucc_status_t free(void* request) = 0;
};

}  // namespace legate::detail::comm::coll
