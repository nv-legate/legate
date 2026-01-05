..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. _ch_tasks:

=====
Tasks
=====

The primary execution unit in a Legate program is a *task*. All interesting operations
performed by a Legate program should be performed within a task. Notably, tasks are also
the only place where the user is able to modify the underlying data of any stores.

Tasks are created through the runtime, filled with their respective arguments and
execution properties, then submitted back to the runtime for execution.

.. _sec_tasks_execution_model:

Execution Model
---------------

A Legate program is split into two parts: the "top-level", and the "leaf task". Top-level
refers to the code that is called directly by the ``main()`` function, i.e. the one which
creates and submits tasks for execution. Leaf-task refers to the actual executed tasks
themselves.

.. note::

   It is only possible to create and submit tasks from within the top-level program.


There are a number of important properties of leaf task execution:

#. Leaf tasks are assumed to execute completely asynchronously from the top-level program.
#. Leaf tasks will not begin execution until all pre-requisites for the task are satisfied.
#. There is no guarantee *where* on the machine the leaf task will be executed. For
   example, the user cannot assume that the task is executed on the same core, CPU, node,
   or (in the case of networked machines) the same physical machine as the code submitting
   the task for execution.
#. There is no method to directly determine whether a leaf task has completed. The only
   reliable methods of determining this are indirect:

   #. By issuing a blocking runtime execution fence (via
      ``legate::Runtime::issue_execution_fence()``) from the top-level. If the blocking
      variant is called, the function will not return until all outstanding tasks have
      completed, and their side-effects are visible.
   #. By executing a dependent leaf task. If task B has inputs which depend on the outputs
      of task A, then if control reaches the *body* of task B, it is guaranteed that task
      A has completed. This is analogous to the pre-requisites requirement above.


.. _sec_tasks_task_ids:

Task IDs
--------

Each task is referenced through a unique identifier, the "task ID" (of type
``legate::LocalTaskID``).  Note that this number is only unique within a ``Library`` once
a task has been registered with it (e.g. there can be multiple tasks with ``LocalTaskID``
0, but under different libraries). The ``Library`` is, in effect, a "name space" for
tasks. This allows libraries to keep an enumeration of their tasks without concern of ID
collision in a multi-library application.

Each task is also allocated a globally unique identifier ``legate::GlobalTaskID``, but
users are discouraged from using this directly.

.. _sec_tasks_task_variants:

Task Variants
-------------

One of the primary abstractions in Legate tasks is the concept of task "variants". In a
nutshell, these allow tasks to communicate which kinds of devices the task is equipped to
run on. Tasks may have multiple variants, but each variant may only be declared
once. Additionally, tasks must have at least one variant.

For example, if the task has a CPU variant, this informs the runtime that such a task is
able to run on any available CPU. If the task also has a GPU variant, this tells the
runtime that the task is also able to execute GPU code.

.. note::

   The task bodies themselves are *always* executed on the CPU. The variant merely informs
   the runtime that the task intends to use particular hardware. For example, the GPU
   variant tells the runtime that the task intends to launch GPU kernels, and/or intends
   to make calls into GPU-accelerated third-party libraries.


The runtime is responsible for selecting which of the available task variants is
executed. Unless otherwise constrained, the runtime will prefer the most "accelerated"
variant whenever possible. For example, if the task has a GPU, OpenMP, and CPU variant,
the runtime will prefer the GPU variant, then the OpenMP, and finally the CPU variant.

The user can manually restrict the variant selection by constructing a ``legate::Scope``
class with an appropriately set ``legate::mapping::Machine``. In this mode, the runtime
will consider only the subset of the machine as specified in the scope, but will still
apply the same variant ranking scheme. For example, assuming an initial machine that
supports GPU, OpenMP, and CPU tasks, and the following scope:

.. code-block:: cpp

   #include <legate.h>

   const auto machine = legate::get_machine();
   const auto scope = legate::Scope{
     machine.only({legate::TaskTarget::OMP, leate::TaskTarget::CPU})
   };

   // create some tasks and submit them...


The runtime will select the OpenMP variant (assuming the task has one), since that target
has a higher variant priority than CPU.

.. note::

   Task variant selection happens when the task object is *constructed*. Any manual
   restriction using ``legate::Scope`` or the like must be done before the call to
   ``legate::Runtime::create_task()``, otherwise the scope has no effect.


.. _sec_tasks_parallelization:

Task Parallelization
--------------------

.. _sec_tasks_declaration:

Declaration
-----------

Legate tasks are declared by defining a C++ task, which publicly inherits from the
``legate::LegateTask`` helper. A minimal example of a task declaration is as follows:

.. code-block:: cpp

   #include <legate.h>

   class MyTask : public legate::LegateTask<MyTask> {
   public:
     static inline const auto TASK_CONFIG = legate:TaskConfig{
       legate::LocalTaskID{0}
     };

     static void cpu_variant(legate::TaskContext ctx);
   };


.. note::

   Even though tasks are declared as C++ classes, this is misleading. **No instance of the
   class is ever constructed**. The class has no ownership semantics, and is functionally
   equivalent to a namespace.


This declares a task type ``MyTask``, which has a local task ID of ``0``, and which has a
"CPU variant". This indicates to Legate the following properties:

#. When this task is registered with a ``Library``, its ``Library``-local ID will be
   ``0`` (derived from ``TASK_CONFIG.task_id()``).
#. It supports execution on CPU's, but does not support execution on other processor
   kinds.

The user is able to specify additional configuration and options for the tasks via the
``TASK_CONFIG`` static member. While optional, the user is *highly* encouraged to fill out
as much of the ``TASK_CONFIG`` as possible, as doing so allows the runtime to make more
optimal decisions when launching the tasks. For example, the user may inform the runtime
of the expected task signature:

.. code-block:: cpp

   #include <legate.h>

   class MyTask : public legate::LegateTask<MyTask> {
   public:
     static inline const auto TASK_CONFIG = legate:TaskConfig{
       legate::LocalTaskID{0}
     }.with_signature(
       legate::TaskSignature{}
         .inputs(2)
         .outputs(2)
         .constraints({
           legate::align(legate::proxy::inputs),
           legate::align(legate::proxy::outputs)
         })
     ).with_variant_options(
       legate::VariantOptions{}
         .with_may_throw_exception(true)
         .with_concurrent(true)
     );

     static void cpu_variant(legate::TaskContext ctx);
   };


This declaration informs the runtime of the following properties for ``MyTask``:

#. The task takes exactly 2 input arguments, and 2 output arguments.
#. The input arguments have an alignment constraint applied to them all, as do the output
   arguments.
#. Furthermore, each variant (if the task had multiple) may potentially throw an
   exception, and requires "concurrent" execution (see
   ``legate::VariantOptions::concurrent`` for reference).

If a task declares its task signature in this manner, then the runtime may be able to more
efficiently lay out the task arguments during task launch. Crucially, the runtime will
also be able to "type check" the task signature at launch. For example, if such a task was
mistakenly launched with only 1 input argument, the runtime would be able to catch this
error at the launch-site.

See ``legate::TaskConfig``, and ``legate::LegateTask`` for further discussion on the
available options and semantics.

.. _sec_tasks_registration:

Registration
------------

After declaring a task, the user must also *register* the task with the Legate
runtime. This registration process makes the runtime aware of the task and "finalizes" the
task in the eyes of the runtime. Any further modification of the task (such as modifying
the ``TASK_CONFIG``) will be ignored after registration.

Each task must be registered with a particular ``legate::Library`` (but can be registered
with any number of ``legate::Library`` objects) before use.

This registration process is made simple via deriving from ``legate::LegateTask``, which
defines helper routines that perform all of the boilerplate registration code for the
user. Registration of tasks may be as simple as:

.. code-block:: cpp

   #include <legate.h>

   auto lib = legate::Runtime::get_runtime()->find_library("my_library");
   MyTask::register_variants(lib);

After the call to ``register_variants()``, the task may now be constructed and launched,
as detailed in the following section.

.. _sec_tasks_launching:

Launching
---------

Once declared and registered, tasks are created via the runtime. This is performed through
the various overloads of ``legate::Runtime::create_task()``. This routine will create
either an ``legate::AutoTask`` or ``legate::ManualTask`` (see
:ref:`sec_tasks_parallelization` for further discussion on the differences between these
classes).

The tasks must be created using the same ``legate::Library`` that it was registered with
(see :ref:`sec_tasks_registration`). Attempting to create a task with a library with
which it was not registered is diagnosed at runtime.

Once created, the task objects are configured with the various arguments and settings
needed to properly launch them. However, as described in :ref:`sec_tasks_declaration`,
many of these can be statically declared in the task body. Usually, the user need only
supply the actual input/output/scalar/etc. arguments at the task launch site:

.. code-block:: cpp

   #include <legate.h>

   void launch_my_task(
     const legate::LogicalArray& input_array,
     // ...
     const legate::LogicalArray& output_array,
     // ...
   )
   {
     auto runtime = legate::Runtime::get_runtime();
     auto lib = runtime->find_library("my_library");
     auto task = runtime->create_task(lib, MyTask::TASK_CONFIG.task_id());

     task.add_input(input_array);
     // ...
     task.add_output(output_array);
     // ...

     runtime->submit(std::move(task));
   }


.. warning::

   Task objects are single-use only. Once submitted (via ``legate::Runtime::submit()``),
   these objects may not be reused to submit a new task. In order to submit a new task of
   the same type, a fresh task object must be constructed.


Due to the deferred nature of the Legate runtime, submission of the task to the runtime
does **not** imply the task has been immediately executed. It simply appends the task to
the Legate launch pipeline to be executed at some unspecified later time. The user has no
guarantee of execution of the task until an explicit, blocking, execution fence is
invoked. See :ref:`ch_runtime` for further discussion of these concepts.
