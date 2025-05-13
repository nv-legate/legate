.. _sec_overview_nutshell:

Legate In A Nutshell
====================

Legate is a task-based programming model. In broad strokes, Legate takes a normal C/C++
program, and asks the user to apply the following transformations:

#. Functions -> Tasks
#. By-value arguments and data -> ``Scalar``
#. By-reference arguments and data -> Various kinds of ``Store``

You effectively tell Legate:

#. Tasks: here's what functions I run.
#. Stores/Scalars: here's what data they touch.
#. Marking task arguments as input/output: And here's how they touch it.

These allow the Legate runtime to abstractly reason about the control and data flow of the
program.

Logical Data Objects
--------------------

- ``LogicalStore`` - An abstract handle to a dense collection of generic, *mutable* data.
- ``LogicalArray`` - A grouping of one or more ``LogicalStore`` s that must be partitioned
  in a related way (e.g. a data store accompanied by a null mask).
- ``Scalar`` - An abstract handle to generic, *immutable* data.

Runtime Objects
---------------

- ``Runtime`` - The "god object" of Legate which controls the entire execution. Almost all
  other objects are created through this.
- ``Library`` - An object to which tasks are registered. Effectively a namespace for tasks
  and their options.
