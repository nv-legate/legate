.. _label_tasks:

.. currentmodule:: legate.core.task

Python Tasks
============

In addition to C++ tasks, Legate allows users to write tasks in pure Python.

Basic Usage
-----------

Tasks are declared by applying the ``legate.task.task`` decorator to a given Python
function. The decorator will then parse the function's signature and register the task
with the runtime. Tasks are executed by calling the function as you would normally. For
example:

.. testcode::

   from legate.core.task import task

   @task # registers the task
   def foo() -> None:
       print("Hello World!")

   foo() # executes the task

.. testoutput::

   "Hello World!"

There are several key restrictions placed on the signature of the task function, all of
which are checked by the decorator.

#. All arguments must have type-hints, without exception.
#. Store arguments must be given as either ``InputStore``, ``OutputStore``,
   ``InputArray``, or ``OutputArray``. Bare ``PhysicalStore`` or ``PhysicalArray``
   arguments are not allowed.
#. The return value of the function must be exactly ``None``. In the future, this
   restriction may be lifted.

It is possible to take and pass store arguments to a Python task, just like a regular C++
task. This is done by using the special ``Input``, ``Ouput``, or ``Reduction`` store/array
type hints:

.. testcode::

   import numpy as np
   from legate.core import get_legate_runtime, types as ty
   from legate.core.task import task, InputArray, OutputArray

   def make_store(init: list[int]):
       arr = np.array(init, dtype=np.int64)
       return get_legate_runtime().create_store_from_buffer(ty.int64, arr.shape, arr, False)

   @task
   def foo_in_out(in_store: InputArray, out_store: OutputArray) -> None:
       # (2)
       in_store = np.asarray(in_store)
       out_store = np.asarray(out_store)
       out_store[:] = in_store[:]


   # (1)
   in_store = make_store([1, 2, 3])
   out_store = make_store([4, 5, 6])
   foo_in_out(in_store, out_store)

   print(np.asarray(out_store.get_physical_store()))

.. testoutput::

   array([1, 2, 3])


An important point to note: at point ``(1)``, the store objects are ``LogicalStore`` (or
``LogicalArray``), but inside the task body (point ``(2)``), they will have automatically
been partitioned across all instances of the task, and transformed to ``PhysicalStore``
(or ``PhysicalArray``), just like in C++ tasks. It is illegal to pass a ``PhysicalStore``
or array as an argument to the function. This is checked on task function call (before it
is launched).

It is also possible to pass and receive arbitrary Python data types as task arguments. Any
arguments passed in this manner should be considered as constant, scalar values
*independent* of the call-site values. Any modifications made to them in the task body may
not propagate outside the task body!

.. testcode::

   from legate.core.task import task

   class MyClass:
       pass

   @task
   def foo(
       x: float,  # basic arithmetic types supported
       y: dict[str, str],  # complex collections as well
       z: MyClass  # and even custom classes
   ) -> None:
       ...


   foo(12.34, {"hello": "there"}, MyClass())


Misc. Trivia
^^^^^^^^^^^^

#. Keyword arguments are supported

.. testcode::

   import numpy as np
   from legate.core import get_legate_runtime, types as ty
   from legate.core.task import task, InputStore, OutputStore

   def make_store(*args, **kwargs):
       arr = np.array([1, 2, 3], dtype=np.int64)
       return get_legate_runtime().create_store_from_buffer(ty.int64, arr.shape, arr, False)


   x_store = make_store()
   y_store = make_store()
   z_store = make_store()


   @task
   def foo(x: InputStore, y: OutputStore, z: OutputStore) -> None:
       ...

   foo(z=z_store, x=x_store, y=y_store) # will demux to f(x_store, y_store, z_store)

#. Default values for arguments are currently not supported.
#. The task may raise arbitrary exceptions, provided that the decorator is passed the
   ``throws_exception=True`` argument, and that the exception derives from
   ``Exception``. If ``throws_exception`` is ``False`` (the default, if not given) then
   Legate will abort when the exception is thrown. If the thrown exception does not derive
   from ``Exception``, behavior is undefined.

.. testcode::

   from legate.core.task import task

   class MyException(Exception):
       pass


   @task(throws_exception=True)
   def foo() -> None:
       raise MyException("exceptional!")


   try:
       foo()
   except MyException as exn:
       print(exn)


.. testoutput::

   "exceptional!"


Task Decorator
--------------

.. autosummary::
   :toctree: generated/

   task

Special Types
-------------

.. autosummary::
   :toctree: generated/

   InputStore
   OutputStore
   ReductionStore
   InputArray
   OutputArray
   ReductionArray

PyTask
------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   PyTask


Variant Invoker
---------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   VariantInvoker
