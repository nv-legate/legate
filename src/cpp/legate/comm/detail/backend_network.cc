/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/comm/detail/backend_network.h>

#include <legate_defines.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/macros.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace legate::detail::comm::coll {

namespace {

std::unique_ptr<BackendNetwork> the_backend_network{};

}  // namespace

/* static */ void BackendNetwork::create_network(std::unique_ptr<BackendNetwork>&& network)
{
  the_backend_network = std::move(network);
}

/* static */ std::unique_ptr<BackendNetwork>& BackendNetwork::get_network()
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG) && LEGATE_UNLIKELY(!BackendNetwork::has_network())) {
    throw legate::detail::TracedException<std::logic_error>{
      "Trying to retrieve backend network before it has been initialized. Call "
      "BackendNetwork::create_network() first"};
  }
  return the_backend_network;
}

/* static */ bool BackendNetwork::has_network() { return the_backend_network != nullptr; }

/* static */ legate::comm::coll::CollCommType BackendNetwork::guess_comm_type_()
{
  // This function is a complete and total HACK, and is needed for
  // comm::cpu::register_tasks(). That function needs to query the comm type (so that it can
  // either register CPU tasks or MPI tasks), so it ideally would check
  // BackendNetwork::get_network()->comm_type.
  //
  // HOWEVER, it is called before comm::coll::init(), and so BackendNetwork is not yet
  // initialized. So we duplicate the selection logic of comm::coll::init() here...
  //
  // We use a static local to cache our first answer (wherever it was called from) so that when
  // we do get around to comm:coll::init(), we can check that the actually created communicator
  // is of the same type that we guessed it would be previously.
  static const auto guessed_comm_type = [] {
    if (LEGATE_DEFINED(LEGATE_USE_NETWORK) && LEGATE_NEED_NETWORK.get(/* default_value */ false)) {
      return legate::comm::coll::CollCommType::CollMPI;
    }
    return legate::comm::coll::CollCommType::CollLocal;
  }();

  return guessed_comm_type;
}

// ==========================================================================================

void BackendNetwork::abort()
{
  // does nothing by default
}

std::int32_t BackendNetwork::get_unique_id_() { return current_unique_id_++; }

void* BackendNetwork::allocate_inplace_buffer_(const void* recvbuf, std::size_t size)
{
  LEGATE_ASSERT(size);

  auto sendbuf_tmp = std::unique_ptr<char[]>{new char[size]};

  std::memcpy(sendbuf_tmp.get(), recvbuf, size);
  return sendbuf_tmp.release();
}

void BackendNetwork::delete_inplace_buffer_(void* recvbuf, std::size_t)
{
  delete[] static_cast<char*>(recvbuf);
}

}  // namespace legate::detail::comm::coll
