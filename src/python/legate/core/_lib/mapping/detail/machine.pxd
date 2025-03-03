# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


cdef extern from "legate/mapping/detail/machine.h" namespace "legate::mapping::detail" nogil:  # noqa
    cdef cppclass _MachineImpl "legate::mapping::detail::Machine":
        pass
