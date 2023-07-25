<!--
SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# Legate IO example

This tutorial illustrates how one can build an I/O library in Legate using the
following three scenarios:

1. I/O with a single file (`write_file`, `read_file`, `read_file_parallel`)
2. I/O with a dataset of uneven tiles (`write_uneven_tiles`, `read_uneven_tiles`)
3. I/O with a dataset of even tiles (`write_even_tiles`, `read_even_tiles`)

This tutorial also teaches you how to make a domain library container interoperate
with other libraries via Legate data interface.
