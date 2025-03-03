# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0


cdef extern from "legate/data/detail/logical_store.h" namespace "legate::detail" nogil:  # noqa E501
    cdef cppclass _LogicalStoreImpl "legate::detail::LogicalStore":
        void allow_out_of_order_destruction()
