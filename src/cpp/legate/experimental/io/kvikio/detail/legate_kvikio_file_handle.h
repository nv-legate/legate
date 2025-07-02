/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// We rename "kvikio" to legate_kvikio to rename all symbols to a unique name for us
// this avoids accidental re-export of kvikio symbosl that may be used by others
// (sure there are other ways to avoid this, but they are harder).
// kvikio uses kvikio:: extensively, so patching seemed not great either.
// See https://github.com/nv-legate/legate.internal/pull/2431
#define kvikio legate_kvikio
#include <kvikio/file_handle.hpp>
#undef kvikio
