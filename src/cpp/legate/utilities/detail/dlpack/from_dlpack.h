/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/external_allocation.h>
#include <legate/data/logical_store.h>

#include <optional>

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

/**
 * @brief Construct an `ExternalAllocation` from a DLPack versioned tensor.
 *
 * This routine takes ownership of `dlm_tensor`, and is responsible for disposing of
 * `dlm_tensor` via the deleter member thereof.
 *
 * If any exception is thrown, the `dlm_tensor` pointer is deleted using the deleter.
 *
 * On success, `dlm_tensor` is set to `nullptr` to indicate that it has been consumed.
 *
 * @param dlm_tensor A pointer (to pointer) of the DLPack tensor.
 * @param read_only Overrides the capsule's read-only flag when provided.
 *
 * @return The external allocation.
 */
[[nodiscard]] LEGATE_PYTHON_EXPORT legate::ExternalAllocation make_external_alloc_from_dlpack(
  DLManagedTensorVersioned** dlm_tensor, std::optional<bool> read_only = std::nullopt);

/**
 * @brief Construct an `ExternalAllocation` from a DLPack un-versioned tensor.
 *
 * This routine has identical ownership semantics as
 * `make_external_alloc_from_dlpack(DLManagedTensorVersioned**)`. See that routine for further
 * discussion on lifetimes.
 *
 * As with the other overload, `dlm_tensor` is set to `nullptr` on success.
 *
 * @note The user should prefer using the versioned DLPack interface instead. There are no
 * protections for backwards incompatibilities with this routine.
 *
 * @param dlm_tensor A pointer (to pointer) of the DLPack tensor.
 * @param read_only Overrides the read-only flag, which defaults to `false` for the
 * unversioned protocol.
 *
 * @return The external allocation.
 */
[[nodiscard]] LEGATE_PYTHON_EXPORT legate::ExternalAllocation make_external_alloc_from_dlpack(
  DLManagedTensor** dlm_tensor, std::optional<bool> read_only = std::nullopt);

}  // namespace legate::detail
