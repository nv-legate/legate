/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/cuda/detail/cuda_util.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/utilities/scope_guard.h>

namespace legate::cuda::detail {

void sync_current_ctx()
{
  auto&& api = get_cuda_driver_api();
  auto event = api->event_create();

  LEGATE_SCOPE_GUARD(api->event_destroy(&event));
  api->ctx_record_event(api->ctx_get_current(), event);

  do {
    const auto result = api->event_query(event);

    if (result == LEGATE_CUDA_ERROR_NOT_READY) {
      continue;
    }

    if (result == LEGATE_CUDA_SUCCESS) {
      break;
    }

    cuda::detail::throw_cuda_driver_error(
      result, /*expression=*/"api->event_query(event)", __FILE__, __func__, __LINE__);
  } while (true);
}

void stream_synchronize_minimal(CUstream stream)
{
  auto&& api = get_cuda_driver_api();
  auto event = api->event_create();

  LEGATE_SCOPE_GUARD(api->event_destroy(&event));

  // Synchronize on the null stream
  api->event_record(event, stream);

  do {
    const auto result = api->event_query(event);

    if (result == LEGATE_CUDA_ERROR_NOT_READY) {
      continue;
    }

    if (result == LEGATE_CUDA_SUCCESS) {
      break;
    }

    cuda::detail::throw_cuda_driver_error(
      result, /*expression=*/"api->event_query(event)", __FILE__, __func__, __LINE__);
  } while (true);
}

}  // namespace legate::cuda::detail
