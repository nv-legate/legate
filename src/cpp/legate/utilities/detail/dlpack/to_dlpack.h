/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/cuda/detail/cuda_driver_types.h>
#include <legate/utilities/detail/dlpack/dlpack.h>

#include <memory>
#include <optional>

namespace legate {

class PhysicalStore;

}  // namespace legate

namespace legate::detail {

/**
 * @brief Export a store into the DLPack format.
 *
 * The value of `copy` has the following semantics:
 *
 * - `true`: Legate *must* copy the data. If Legate fails to copy the data, for any reason,
 *   an exception is thrown.
 * - `false`: Legate must *never* copy the data. If the store cannot be exported without a
 *   copy, then an exception is thrown.
 * - `std::nullopt`: Legate may copy the data if it is deemed necessary. Currently, this is
 *   never the case, and Legate will always provide a view.
 *
 * In any case, if a copy is made, the `DLManagedTensorVersioned::flags` member will have the
 * `DLPACK_FLAG_BITMASK_IS_COPIED` bit set.
 *
 * If `max_version` is provided, and this routine cannot satisfy the version requirement, an
 * exception is thrown. If `max_version` is not provided, then the returned tensor will be of
 * the highest version available.
 *
 * Since `PhysicalStore`s cannot be moved, `device` has very little use (and exists purely to
 * allow the Python API to pass it, per the array standard). In a nutshell, if the device of
 * the tensor that would be created by this routine doesn't exactly match `device`, then an
 * exception is thrown.
 *
 * The `std::unique_ptr` returned by this routine will automatically call the deleter of the
 * DLPack tensor in its destructor.
 *
 * @param store The store to export.
 * @param copy Whether to copy the underlying data or not.
 * @param stream A stream on which the data must be coherent after this routine returns.
 * @param max_version The maximum version that the returned tensor should support.
 * @param device The device that the returned tensor should reside on.
 *
 * @return The DLPack managed tensor.
 */
[[nodiscard]] LEGATE_PYTHON_EXPORT
  std::unique_ptr<DLManagedTensorVersioned, void (*)(DLManagedTensorVersioned*)>
  to_dlpack(const legate::PhysicalStore& store,
            std::optional<bool> copy                 = std::nullopt,
            std::optional<CUstream> stream           = std::nullopt,
            std::optional<DLPackVersion> max_version = std::nullopt,
            std::optional<DLDevice> device           = std::nullopt);

}  // namespace legate::detail
