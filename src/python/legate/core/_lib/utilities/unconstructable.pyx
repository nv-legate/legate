# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cdef class Unconstructable:
    def __init__(self, *args, **kwargs) -> None:
        raise ValueError(
            f"{type(self).__name__} objects must not be constructed directly"
        )
