.. _ch_data:

==========
Data Model
==========

.. _sec_data_logical_stores:

Logical Stores
==============

When the top-level program instructs the runtime to create a store, the runtime will
return a "logical" store handle. Logical stores are distinct from "physical" stores, which
are discussed in :any:`sec_data_physical_stores`.

These logical handles are abstract views over the underlying data. The user is able to
manipulate and reshape these views at will in order to extract or modify various pieces of
a store.

.. important::

   Logical stores are not "parallel" objects, and have no concept of "local" or "remote"
   sizes. They always refer to the *entire* storage, no matter how many ranks the program
   is run under. For example, some libraries might allow you to create parallel vectors,
   but each rank owns only a subset of that vector. There is no such distinction in
   Legate. Each rank will get an equivalent handle to the full "global" size of the store.


Logical stores are fundamentally *lazy* objects. Obtaining a logical store does not
necessarily represent the allocation of any actual memory:

When a store is created, the user effectively informs the runtime "I would like *some*
kind of buffer, with this particular shape to exist". The runtime has no obligation to
materialize the buffer in any way until such time as the buffer is observable by the user.

In the vast majority of cases, a store is considered "observable" (and therefore
materialized) only when the store is accessed by a task. This could be by requesting to
read the buffer (adding it as a input parameter), but could also be from write-only access
(adding the store as an output parameter), which also causes the pieces to be
materialized.

.. code-block:: cpp
   :dedent:

   #include <legate.h>

   class SomeTask : public LegateTask<SomeTask> {
   public:
     static constexpr auto TASK_ID = ...;

     static void cpu_variant(legate::TaskContext ctx)
     {
       // Only when control reaches this point will the store have been materialized
     }
   };

   auto runtime = legate::Runtime::get_runtime();

   // Create a handle to an array of 10 int32's. The array at this point contains
   // indeterminate values. It is an error to try and read from it.
   legate::LogicalStore store = runime->create_store(legate::Shape{10}, legate::int32());

   // Initialize the store with some data.
   runtime->issue_fill(store, legate::Scalar{7});

   auto task = runtime->create_task(some_library, SomeTask::TASK_ID);

   // Inform the runtime that we intend to read from the store. The runtime still won't
   // materialize the store at this point (simply creating and filling the task does not
   // count as "observing" it).
   task.add_input(store);
   // Submitting the task still does not guarantee that the store has been materialized.
   runtime->submit(std::move(task));


This property also applies to modifications on logical stores. Note that when we
initialize the store via the ``Runtime::issue_fill()`` call, nothing is actually done at
that point. The runtime simply notes that this operation is requested, and appends it to a
list of logical operations.

The fill would not be executed until right before control is transferred to the task
body. An interesting point here is that if we amend the example to this time add the store
as an *output*:

.. code-block:: cpp
   :dedent:

   task.add_output(store);


Then the fill would not be executed at all, because the result of that fill would not be
visible to the user. This only applies if the store is a pure output parameter. Adding it
as both an input and an output:

.. code-block:: cpp
   :dedent:

   task.add_input(store);
   task.add_output(store);

Would still trigger the fill, as the user is requesting to read the old values.

.. _sec_data_physical_stores:

Physical Stores
===============

The flip-side of logical stores are the "physical" stores. Physical stores allow the user
to access the raw underlying buffer and manipulate it. As such, they are usually only
exposed to users inside leaf tasks (see :any:`sec_runtime_execution_model` for more
information on tasks).

.. note::

   As opposed to logical stores, physical stores *are* "parallel" objects. When a logical
   store is passed to a task as an argument, once inside the task, each rank will receive
   a physical store over a local subset of the logical region.
