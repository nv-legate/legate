.. _ch_runtime:

===========
The Runtime
===========

.. _sec_runtime_execution_model:

Execution Model
===============

Tasks are always executed asynchronously with respect to the top-level program. The only
way to observe task execution is by inserting an explicit blocking fence into the program
(via ``legate::Runtime::issue_execution_fence()``).

This asynchronous execution model underpins all modes of operation. The most common
approach, described below, is standard execution, which defines how leaf tasks are
grouped, scheduled, and run under typical conditions.

Standard Execution
------------------

In standard execution mode, leaf-tasks are submitted and scheduled as "blocks" of
tasks. Each leaf task in a block may be scheduled at an arbitrary time and arbitrary
order. For example, if a task is launched that creates 100 leaf tasks, those tasks can
execute at any time so long as each individual task's inputs are satisfied. Other tasks
may also interleave with these leaf tasks, so long as there are no dependencies between
the leaves.

If, however, there is a data dependency between tasks (say, task B depends on task A),
then the runtime will ensure that every leaf of task A has finished executing before the
first leaf of task B begins execution.

The exception to this rule are "collective" tasks. These tasks are executed
concurrently. Note, this concurrency is a *requirement*, not a grant. The entire machine
must execute the tasks at exactly the same time as one giant block. Non-collective tasks
may still execute at the same time, but once a collective task is launched, no other
collective task may execute concurrently. For example, if tasks A and C are collective,
but task B is not (and none of the tasks share any data dependencies that might otherwise
enforce ordering), then the runtime may schedule these tasks as::

  time ->

  A[0] | A[1] | ... | C[0]
  B[0] | B[1] |     | C[1]


But it will never schedule the tasks as::

  time ->

  A[0] | A[1] | ... | B[0]
  C[0] | C[1] |     | ...


Streaming Execution
-------------------

Streaming execution is a special execution mode which aims to reduce the memory pressure
of large parallel tasks. A common problem when working with large datasets is that many
modifications of the data are discarded/overwritten immediately by subsequent tasks. For
example, suppose that we wanted to load two very large vectors (``x`` and ``y``) from
disk, add them together, and store the result in a third vector (``r``), i.e. ``r = x[:] +
y[:]``. Once ``r`` is computed, we don't need ``x`` or ``y`` anymore and can discard
them. Conceptually, for this operation, we do not need to keep ``x``, ``y`` and ``r``
alive for the entire duration of the operation. Instead, we should load a chunk of ``x``
and ``y``, perform the summation on that chunk, and discard (or reuse) the memory during
the next leaf task.

In the standard execution model, this approach is impossible. The standard execution model
would not launch the summation task before both of the load tasks are fully completed (and
therefore causing the full sizes of ``x``, ``y``, and ``r`` to be resident
simultaneously). If the vectors are big enough, Legate would crash with out-of-memory
errors.

Streaming execution takes all the restrictions and rules of standard execution mode, but
relaxes the constraint that *all* leaf tasks must execute before dependent tasks begin. In
other words, consider a set of tasks A, B, C. Each task depends on the previous (C depends
on B, depends on A) and each parallelized into N leaf tasks. This can be modeled as a
matrix of tasks::

  [A[0], A[1], ..., A[N],
   B[0], B[1], ..., B[N],
   C[0], C[1], ..., C[N]]

The standard execution model would schedule these tasks "row-wise", i.e. execute all of A
before executing all of B before executing all of C::

  time ->

  [A[0], A[1], ..., A[N],
   |--------------------x B[0], B[1], ..., B[N],
   |-------------------------------------------x C[0], C[1], ..., C[N]]

The streaming execution model on the other hand would execute "column-wise". It would
select some column of the matrix, and execute every task in that column before executing
the others::

  time ->

  [A[0],
   B[0],
   C[0],
    ||  A[1],
    ||  B[1],
    ||  C[1],
    ...  ...
    ||   ||   A[N],
    ||   ||   B[N],
    ||   ||   C[N]]


In our previous example, this means that we perform exactly the optimization of loading
only a chunk of ``x`` and ``y`` at a time, instead of having them both be resident. In
this case ``A`` would be the load of ``x``, ``B`` would be the load of ``y`` and ``C``
would be the summation.

Streaming execution may be activated by placing a section of code inside a streaming scope

.. code-block:: cpp

   {
     auto scope = legate::Scope{legate::ParallelPolicy{}.with_streaming(true)};

     // These tasks will all have the streaming execution policy applied to them.
     launch_task_A();
     launch_task_B();
     launch_task_C();
   }

   // This task will execute under the standard execution policy.
   launch_task_A();
