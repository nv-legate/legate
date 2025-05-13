.. _ch_mapping:

=======
Mapping
=======

When a task is submitted for execution in Legate, it does not begin running immediately.
Instead, it goes through a series of steps that prepare it for efficient execution on the
underlying hardware. A key step in this process is *mapping*, which determines how and
where the task will run.

Legate divides task execution into two conceptual parts: the *what* and the *how*.

The *what* refers to the logical structure of the program—the tasks that are submitted,
the specific task variants that are selected, the data the tasks operate on, and the
dependencies between them. It also includes the actual implementation of the tasks (their
bodies). This portion of the execution is defined by the application code itself and is
handled by the top-level task.

The *how* is determined by the *mapper*, a component that decides how the logical tasks
are mapped onto the physical machine. The mapper is responsible for selecting the hardware
resources each task should run on (such as which CPU core or node), allocating the
necessary physical memory for task arguments, and deciding how to shard tasks across
resources to enable parallel execution.

By separating the logic of what tasks do from how they are scheduled and placed on
hardware, Legate enables a modular and portable programming model. Tasks can be written
without concern for the underlying machine configuration, making code easier to develop
and maintain.

.. _sec_mapping_exec_pipeline:

Execution Pipeline Overview:
----------------------------

The typical execution flow for a Legate task is as follows:

1. **Task Submission** – The user program submits tasks to the Legate runtime.
2. **Runtime Analysis** – The runtime analyzes task dependencies and prepares the
   execution plan.
3. **Mapping** – The mapper is invoked via a series of callbacks, and assigns tasks to
   hardware resources and manages memory allocation.
4. **Execution** – Once mapped, the runtime executes the tasks on the designated hardware.

This pipeline allows Legate to efficiently schedule and execute large numbers of parallel
tasks while abstracting away low-level system details.

Legate distills the mapping interface down to a core set of decisions, encapsulated by the
``legate::mapping::Mapper`` class. By default, Legate provides a default mapper
implementation which is selected whenever the user does not specify their own mapper. The
default mapper is designed to be reasonably performant in most cases, and users generally
do not need to override it.

The user may however define their own custom mappers by deriving from
``legate::mapping::Mapper``, and passing their derived class as an additional argument
when creating ``Library``'s:

.. code-block:: cpp
   :dedent:

      #include <legate.h>

      class UserMapper : public legate::mapping::Mapper {
        // implement required functions...
      };

      auto my_lib = legate::Runtime::get_runtime().create_library(
        "my_library",
        legate::ResourceConfig{},
        std::make_unique<UserMapper>()
      );


Mapper Callback Model
---------------------

As detailed in :ref:`sec_mapping_exec_pipeline`, mappers are invoked via a series of
callbacks made from the runtime after task submission. Each callback requests a particular
piece of information from the mapper in order to map the tasks onto the hardware.

All mappers in Legate execute within a dedicated, single thread that runs concurrently
with the main application thread (which executes the top-level task). Although only one
mapper callback is active at any given time, multiple callbacks may be interleaved over
the course of execution. This means that callbacks are not executed in parallel, but may
be suspended and resumed at different points, depending on the runtime's scheduling.

As a result, if your mapper maintains shared state or interacts with external resources,
it must be written to handle interleaving safely. The mapper thread runs concurrently with
the main thread, and care must be taken to avoid unintended side effects.

.. warning::

   Legate does not provide any built-in concurrency protection for mappers. If your mapper
   accesses resources shared with the main thread, it is your responsibility to implement
   appropriate synchronization mechanisms, such as mutexes.


While the single-threaded mapper model simplifies some aspects of concurrency, developers
must still ensure that their code is reentrant and properly synchronized to avoid race
conditions or inconsistent behavior.

For this reason, it is also advised that mappers be purely functional, and *not* depend on
additional external state. If a mapper requires state, then it should store that state as
member variables.
