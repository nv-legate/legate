/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/utilities/detail/omp_thread_local_storage.h>

namespace legate::detail {

template <typename VAL>
OMPThreadLocalStorage<VAL>::OMPThreadLocalStorage(std::uint32_t num_threads)
  : storage_(PER_THREAD_SIZE * num_threads)
{
}

template <typename VAL>
VAL& OMPThreadLocalStorage<VAL>::operator[](std::uint32_t idx)
{
  return *reinterpret_cast<VAL*>(storage_.data() + PER_THREAD_SIZE * idx);
}

}  // namespace legate::detail
