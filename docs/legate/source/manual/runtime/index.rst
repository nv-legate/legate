..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. _ch_runtime:

===========
The Runtime
===========

The Legate Runtime is the central component of the library that manages the lifetime of
tasks and data. It provides the users an API to create bulk parallel tasks and data, such
as Logical Arrays and Stores. The implementation of this API provides the capability of
launching tasks on parallel hardware.

.. _sec_runtime_execution_model:

Execution Model
---------------

Tasks are always executed asynchronously with respect to the top-level program. The only
way to observe task execution is by inserting an explicit blocking fence into the program
(via ``legate::Runtime::issue_execution_fence()``).

This asynchronous execution model underpins all modes of operation. The Runtime supports
two modes of task execution which are described below. By default operations are submitted
in their entirety for execution on the parallel hardware in program order. This is called
Standard execution. Streaming mode is an alternate execution mode wherein tasks can be
scheduled in batches (given they satisfy certain conditions).

.. toctree::
   Standard Execution Model <standard_execution.rst>
   Streaming Execution Model <streaming.rst>
