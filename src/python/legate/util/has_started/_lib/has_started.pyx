# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cpdef bool runtime_has_started():
    r"""
    Whether the Legate runtime has been started.

    This function is not in legate.core because importing legate.core will
    automatically start the legate runtime.

    Returns
    -------
    bool
    """
    return has_started()
