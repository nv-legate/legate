.. currentmodule:: legate.core

Shape
=====

A ``Shape`` is used in expressing the shape of a certain entity in Legate. The
reason Legate introduces this indirection to the shape metadata is that stores
in Legate can have unknown shapes at creation time; the shape of an unbound
store is determined only when the producer task finishes. The shape object can
help the runtime query the store's metadata or construct another store
isomorphic to the store without getting blocked.

Shape objects should behave just like an array of integers, but operations that
introspect the values implicitly block on completion of the producer task.


.. autosummary::
   :toctree: generated/

   Shape.__init__


Properties
----------
.. autosummary::
   :toctree: generated/

   Shape.extents
   Shape.ndim
   Shape.volume
   Shape.__getitem__
   Shape.__len__
