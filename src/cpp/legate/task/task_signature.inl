/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/task_signature.h>

namespace legate {

inline const SharedPtr<detail::TaskSignature>& TaskSignature::impl() const { return pimpl_; }

inline SharedPtr<detail::TaskSignature>& TaskSignature::impl_() { return pimpl_; }

}  // namespace legate
