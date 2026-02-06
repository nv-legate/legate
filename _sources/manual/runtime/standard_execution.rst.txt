..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0


Standard Execution
==================

When scheduling a task, the Runtime may further split a task (and its data) into "leaf"
tasks that are scheduled on individual processors in the hardware, such as CPU cores and
GPUs.  In standard execution mode, leaf-tasks are submitted and scheduled as "blocks" of
tasks. Each leaf task in a block may be scheduled at an arbitrary time and arbitrary
order. For example, if a task is launched that creates 100 leaf tasks, those tasks can
execute at any time so long as each individual task's inputs are satisfied. Other tasks
may also interleave with these leaf tasks, so long as there are no dependencies between
the leaves.

If, however, there is a data dependency between tasks (say, task B depends on task A),
then the runtime will ensure that every leaf of task A has finished executing before the
first leaf of task B begins execution.

The exception to this rule are "collective" tasks. These tasks are executed concurrently.
Note, this concurrency is a *requirement*, not a grant. The entire machine must execute
the tasks at exactly the same time as one giant block. Non-collective tasks may still
execute at the same time, but once a collective task is launched, no other collective task
may execute concurrently. For example, if tasks A and C are collective, but task B is not
(and none of the tasks share any data dependencies that might otherwise enforce ordering),
then the runtime may schedule these tasks as::

  time ->

  A[0] | A[1] | ... | C[0]
  B[0] | B[1] |     | C[1]


But it will never schedule the tasks as::

  time ->

  A[0] | A[1] | ... | B[0]
  C[0] | C[1] |     | ...
