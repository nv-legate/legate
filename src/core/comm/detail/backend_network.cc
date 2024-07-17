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

#include "core/comm/detail/backend_network.h"

#include "core/utilities/assert.h"

#include <cstdlib>
#include <cstring>

namespace legate::detail::comm::coll {

void BackendNetwork::abort()
{
  // does nothing by default
}

std::int32_t BackendNetwork::get_unique_id_() { return current_unique_id_++; }

void* BackendNetwork::allocate_inplace_buffer_(const void* recvbuf, std::size_t size)
{
  LEGATE_ASSERT(size);
  void* sendbuf_tmp = std::malloc(size);
  LEGATE_CHECK(sendbuf_tmp != nullptr);
  std::memcpy(sendbuf_tmp, recvbuf, size);
  return sendbuf_tmp;
}

void BackendNetwork::delete_inplace_buffer_(void* recvbuf, std::size_t) { std::free(recvbuf); }

std::unique_ptr<BackendNetwork> backend_network{};

}  // namespace legate::detail::comm::coll
