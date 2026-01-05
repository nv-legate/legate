/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/logical_store.h>

struct DLManagedTensorVersioned;
struct DLManagedTensor;

namespace legate::detail {

/**
 * @brief Construct a LogicalStore from a DLPack versioned tensor.
 *
 * This routine takes ownership of `dlm_tensor`, and is responsible for disposing of
 * `dlm_tensor` via the deleter member thereof.
 *
 * If any exception is thrown, the `dlm_tensor` pointer is deleted using the deleter.
 *
 * On success, `dlm_tensor` is set to `nullptr` to indicate that it has been consumed.
 *
 * @param dlm_tensor A pointer (to pointer) of the DLPack tensor.
 *
 * @return The logical store.
 */
[[nodiscard]] LEGATE_PYTHON_EXPORT legate::LogicalStore from_dlpack(
  DLManagedTensorVersioned** dlm_tensor);

/**
 * @brief Construct a LogicalStore from a DLPack un-versioned tensor.
 *
 * This routine has identical ownership semantics as
 * `from_dlpack(DLManagedTensorVersioned**)`. See that routine for further discussion on
 * lifetimes.
 *
 * As with the other overload, `dlm_tensor` is set to `nullptr` on success.
 *
 * @note The user should prefer using the versioned DLPack interface instead. There are no
 * protections for backwards incompatibilities with this routine.
 *
 * @param dlm_tensor A pointer (to pointer) of the DLPack tensor.
 *
 * @return The logical store.
 */
[[nodiscard]] LEGATE_PYTHON_EXPORT legate::LogicalStore from_dlpack(DLManagedTensor** dlm_tensor);

}  // namespace legate::detail
