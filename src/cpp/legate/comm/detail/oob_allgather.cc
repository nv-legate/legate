/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/oob_allgather.h>

namespace legate::detail::comm::coll {

ucc_status_t OOBAllgather::oob_allgather(
  void* send_buf, void* recv_buf, std::size_t size, void* allgather_info, void** request) noexcept
{
  // allgather_info is the pointer to the OOBAllgather instance, we can cast it to the correct
  // type
  auto* impl = static_cast<OOBAllgather*>(allgather_info);

  *request = impl;
  return impl->allgather(send_buf, recv_buf, size, allgather_info, request);
}

ucc_status_t OOBAllgather::oob_free(void* request) noexcept
{
  auto* impl = static_cast<OOBAllgather*>(request);

  return impl->free(request);
}

ucc_status_t OOBAllgather::oob_test(void* request) noexcept
{
  auto* impl = static_cast<OOBAllgather*>(request);

  return impl->test(request);
}

}  // namespace legate::detail::comm::coll
