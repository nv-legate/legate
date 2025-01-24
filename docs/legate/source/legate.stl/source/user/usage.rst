.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: LicenseRef-NvidiaProprietary

   NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
   property and proprietary rights in and to this material, related
   documentation and any modifications thereto. Any use, reproduction,
   disclosure or distribution of this material and related documentation
   without an express license agreement from NVIDIA CORPORATION or
   its affiliates is strictly prohibited.

Usage
=====

Initializing the Legate runtime
-------------------------------

Before using Legate.STL, you must initialize the Legate runtime. Legate.STL
provides a convenience class called
:cpp:class:`legate::experimental::stl::initialize_library` that initializes the
runtime when it is constructed and finalizes the runtime when it is destroyed.
The runtime must be initialized before any other Legate.STL functions are
called.

.. code-block:: c++

   #include <legate/experimental/stl.hpp>

   namespace stl = legate::experimental::stl;

   int main(int argc, char* argv[]) {
     // Initialize the Legate runtime:
     stl::initialize_library init{argc, argv};

     // Your code here...

     // Finalize the Legate runtime:
     return 0;
   }


Declaring and initializing a store
----------------------------------

There are several ways to declare and initialize a store. The simplest is to
construct an :cpp:class:`legate::experimental::stl::logical_store` object, specifying the element type and
and the shape of the store:

.. code-block:: c++

   // Declare an uninitialized 2-D store:
   stl::logical_store<int, 2> store{{100, 100}};

The code above declares an uninitialized store of ``int``'s with 100 rows and 100
columns. The second template parameter (``2``) specifies the number of dimensions
for the store.

To initialize the store with a value, pass the value as the second argument to
the constructor:

.. code-block:: c++

   // Declare a 2-D store that is zero-initialized:
   stl::logical_store<int, 2> store{{100, 100}, 0};

With this form of initialization, the template parameters can be deduced. This
definition is equivalent to the one above:

.. code-block:: c++

   // Declare a 2-D store that is zero-initialized:
   stl::logical_store store{{100, 100}, 0};

Above, the dimensionality of the store is deduced from the number of elements
in the shape vector (``{100, 100}``). The element type is deduced from the
second argument (``0``).

.. note::

   Declaring a store with an initial value is equivalent to declaring an
   uninitialized store and then calling the :cpp:func:`legate::experimental::stl::fill`
   algorithm.

Finally, you can create a store *in situ* using the
:cpp:func:`legate::experimental::stl::create_store` function:

.. code-block:: c++

   // Declare a 2-D store that is zero-initialized:
   auto store = stl::create_store<int, 2>({100, 100}, 0);

   // Same:
   auto store = stl::create_store({100, 100}, 0);

Scalar stores
.............

There is a special case of a store called a "scalar store." A scalar store is a
store with zero dimensions. It is a store that holds a single element. You can
declare a scalar store like this:

.. code-block:: c++

   // Declare a scalar store that is zero-initialized:
   stl::logical_store<int, 0> store{{}, 0};

   // Same:
   auto store = stl::create_store<int, 0>({}, 0);

Just as with higher-dimensional stores, the element of the store can be
accessed by asking the store for an ``mdspan`` view:

.. code-block:: c++

   // Get a view of the store:
   auto view = stl::as_mdspan(store);

   // Access the element of the store. This is a 0-D indexing operation:
   int i = view();

Due to a limitation in the current implementation of Legate.STL, scalar stores
are immutable. Attempts to modify the value of a scalar store will result in a
runtime error.

There is a convenience function called :cpp:func:`legate::experimental::stl::scalar`
for creating scalar stores:

.. code-block:: c++

   // Declare a scalar store holding the value 0:
   auto store = stl::scalar(0);

   // Same:
   auto store = stl::scalar<int>(0);

   // Same:
   auto store = stl::create_store<int, 0>({}, 0);

.. _element-types:

Element types
.............

The permissible element types for ``logical_store`` objects are:

   - ``__half``
   - ``float``
   - ``double``
   - ``std::int8_t``
   - ``std::int16_t``
   - ``std::int32_t``
   - ``std::int64_t``
   - ``std::uint8_t``
   - ``std::uint16_t``
   - ``std::uint32_t``
   - ``std::uint64_t``
   - ``bool``
   - ``std::complex<float>``
   - ``std::complex<double>``
   - ``std::string``


Accessing the elements of a store
---------------------------------

A store is a logical entity that may span many disjoint address spaces, possibly
distributed across multiple nodes. Unlike an ordinary STL container, the
individual elements cannot be directly referenced.

To access the elements of a store, you must first obtain a view into the store.
There are several ways to do this, but the easiest is to use the
:cpp:func:`legate::experimental::stl::as_mdspan` function. This function returns a
`std::mdspan <https://en.cppreference.com/w/cpp/container/mdspan>`_ object that
gives direct access to the physical elements of the store.

.. code-block:: c++

   // Declare a 2-D store that is zero-initialized:
   stl::logical_store<int, 2> store{{100, 100}, 0};

   // Get a view of the whole the store:
   auto view = stl::as_mdspan(store);

   // Access the elements of the store:
   const auto [imax, jmax] = store.extents();
   for (int i = 0; i < imax; ++i)
     for (int j = 0; j < jmax; ++j)
       view(i, j) = i + j;

.. warning::

   The ``as_mdspan`` function must pull all of the store's backing data onto a
   single node and reify it in physical memory. This can be a very expensive
   operation. The call to ``as_mdspan`` blocks until all the data is available.

There are other ways to obtain a view into a store. See the section on `Creating
Views`_ for more information.

Using algorithms
----------------

Legate.STL provides a number of algorithms that operate on stores. These
algorithms are similar to the range-based algorithms in the C++20 Standard
Library, but they are designed to operate on stores and views of stores rather
than STL ranges.

.. code-block:: c++

   // Declare a 2-D store that is zero-initialized:
   stl::logical_store<int, 2> store{{100, 100}, 0};

   // Assign a value to all elements of the store
   // with the `fill` algorithm:
   stl::fill(store, 42);

   // Get a view into the store:
   auto view = stl::as_mdspan(store);

   // Access the elements of the store:
   const auto [imax, jmax] = store.extents();
   for (int i = 0; i < imax; ++i)
     for (int j = 0; j < jmax; ++j)
       assert(view(i, j) == 42);


Reductions
..........

Reductions are algorithms on stores that reduces the stores' dimensionality by
combining elements along a chosen axis via repeated application of a binary
operation. In the Standard Template Library, the ``std::reduce`` and
``std::accumulate`` algorithms are both examples of reductions.

In Legate, reductions are a complicated topic. Legate.STL makes working
with reductions easier by providing a familiar interface and hiding the
bookkeeping details. The :cpp:func:`legate::experimental::stl::reduce` and
:cpp:func:`legate::experimental::stl::transform_reduce` algorithms, together
with the views described below, let you apply reductions along any axis of a
store using the familiar function objects from the Standard library like
``std::plus`` and ``std::minus``.

Below is an example that does a row-wise reduction of a 2-D store using
using ``std::plus``. It is described after the break.

.. code-block:: c++

      auto store  = stl::create_store<std::int64_t>({3,4});

      // fill the store with data so it looks like this:
      //     [[0 0 0 0]
      //      [1 1 1 1]
      //      [2 2 2 2]]

      auto init   = stl::create_store({4}, std::int64_t{0});
      auto result = stl::reduce(stl::rows_of(store), init, stl::elementwise(std::plus<>()));

      // result is a 1D logical store with the following values:
      // [3 3 3 3]

Like ``std::reduce``, the ``legate::experimental::stl::reduce`` algorithm takes
a range of elements, an initial value, and a binary operation. The code above
used the :cpp:func:`legate::experimental::stl::rows_of` view (described below)
to get a range of rows from the store. The result is a range of 1-D ``mdspan``
objects, each representing a row.

Element-wise operations
.......................

The code above wants to fold a store's rows together using ``std::plus``. Here
we have a problem: ``std::plus`` works on elements, but we want it to work on
*rows*, where each row is an ``mdspan``. You can't add two ``mdspan`` objects
together, and even if you could, assigning the result to another ``mdspan`` is
not going to assign the *elements* of the ``mdspan``. This is where the
:cpp:func:`legate::experimental::stl::elementwise` function comes in. It adapts
a function that works on elements to one that works on ``mdspan``'s of elements.

Passing two ``mdspan`` objects to the result of
``stl::elementwise(std::plus<>)`` will return a new, special "element-wise"
``mdspan``. When you access an element of the element-wise ``mdspan``, it will
call ``std::plus`` on the corresponding elements of the two input ``mdspan``'s.
Legate.STL also knows that assigning an "element-wise" ``mdspan`` to a regular
``mdspan`` should assign the elements of the span rather than the span itself.

Custom reduction operations
...........................

Legate does not recognize the STL functional objects like ``std::plus`` as
reductions. Legate.STL transparently replaces these objects with reduction
operations that Legate understands, like :cpp:class:`legate::SumReduction`.
But Legate.STL's reduction operations are not discriminating; you are free to
use Legate's reduction operations directly if you prefer.

.. Commenting this out until make_reduction is passing its tests.
   You can also use the :cpp:func:`legate::experimental::stl::make_reduction`
   function to build a custom reduction operation. See the reference documentation
   for ``make_reduction`` for more information.

For more algorithm information
..............................

The current set of Legate.STL algorithms is small, but it will grow over time.
It currently includes:

* :cpp:func:`legate::experimental::stl::fill`
* :cpp:func:`legate::experimental::stl::for_each`
* :cpp:func:`legate::experimental::stl::for_each_zip`
* :cpp:func:`legate::experimental::stl::reduce`
* :cpp:func:`legate::experimental::stl::transform`
* :cpp:func:`legate::experimental::stl::transform_reduce`

.. _creating-views:

Creating views
--------------

Legate.STL provides sequence algorithms like ``transform`` and ``reduce`` that,
like their counterparts in the C++20 STL, operate on ranges of elements. There
are many ways to view a store as a range of elements. For example, you may want
to operate to operate on a flattened view of a store, or on a slice of a store.
You may want the elements of the range to be rows or columns or some other
subdimension of the store. For each of these cases, Legate.STL provides a
range adaptor that presents the store as a range of elements.

``elements_of``
...............

The ``elements_of`` adaptor presents the store as a flattened range of elements.
Iterating over the range visits each element of the store in row-major order.

.. code-block:: c++

   // Declare a 2-D store that is zero-initialized:
   stl::logical_store<int, 2> store{{2, 2}, 0};

   // Fill the store with data.
   auto view = stl::as_mdspan(store);
   const auto [imax, jmax] = store.extents();
   for (int i = 0; i < imax; ++i)
     for (int j = 0; j < jmax; ++j)
       view(i, j) = i * imax + j;

   // Use 'elements_of' to get a flattened view of the store:
   auto elements = stl::elements_of(store);
   for (auto& e : elements) {
     std::cout << e << ", ";
   }

The code above prints:

.. code-block:: text

   0, 1, 2, 3,


``rows_of``
...........

The ``rows_of`` adaptor presents a two-dimensional store as a range of rows,
where each row is represented as a 1-dimensional ``stl::logical_store`` object.

.. code-block:: c++

   // Declare a 2-D store that is zero-initialized:
   stl::logical_store<int, 2> store{{2, 2}, 0};

   // Fill the store with data.
   auto view = stl::as_mdspan(store);
   const auto [imax, jmax] = store.extents();
   for (int i = 0; i < imax; ++i)
     for (int j = 0; j < jmax; ++j)
       view(i, j) = i * imax + j;

   // Use 'rows_of' to get a view of the store as a range of rows:
   auto rows = stl::rows_of(store);
   for (stl::logical_store<int, 1> row : rows) {
     auto row_view = stl::as_mdspan(row);
     for (auto i = 0; i < row_view.extent(0); ++i) {
       std::cout << row_view(i) << ", ";
     }
     std::cout << std::endl;
   }

The above code prints:

.. code-block:: text

   0, 1,
   2, 3,

``columns_of``
..............

The ``columns_of`` adaptor presents a two-dimensional store as a range of columns,
where each column is represented as a 1-dimensional ``stl::logical_store`` object.

.. code-block:: c++

   // Declare a 2-D store that is zero-initialized:
   stl::logical_store<int, 2> store{{2, 2}, 0};

   // Fill the store with data.
   auto view = stl::as_mdspan(store);
   const auto [imax, jmax] = store.extents();
   for (int i = 0; i < imax; ++i)
     for (int j = 0; j < jmax; ++j)
       view(i, j) = i * imax + j;

   // Use 'columns_of' to get a view of the store as a range of columns:
   auto cols = stl::columns_of(store);
   for (stl::logical_store<int, 1> col : cols) {
     auto col_view = stl::as_mdspan(col);
     for (auto i = 0; i < col_view.extent(0); ++i) {
       std::cout << col_view(i) << ", ";
     }
     std::cout << std::endl;
   }

The above code prints:

.. code-block:: text

   0, 2,
   1, 3,

``projections_of``
..................

The ``projections_of`` adaptor is a generalization of ``rows_of`` and ``columns_of``.
It presents a store as a range of slices along several specified dimensions. As
such, it can be used with stores of any dimensionality.

``rows_of(store))`` is equivalent to ``projections_of<0>(store)``.

``columns_of(store))`` is equivalent to ``projections_of<1>(store)``.

.. code-block:: c++

   // Declare a 3-D store that is zero-initialized:
   stl::logical_store<int, 3> store{{2, 2, 2}, 0};

   // Fill the store with data.
   auto view = stl::as_mdspan(store);
   const auto [imax, jmax, kmax] = store.extents();
   for (int i = 0; i < imax; ++i)
     for (int j = 0; j < jmax; ++j)
       for (int k = 0; k < kmax; ++k)
         view(i, j, k) = i * imax * jmax + j * jmax + k;

   // Use 'projections_of' to get a view of the store as a range of slices along
   // the first and second dimension:
   auto slices = stl::projections_of<0,1>(store);
   for (stl::logical_store<int, 1> slice : slices) {
     auto slice_view = stl::as_mdspan(slice);
     for (auto i = 0; i < slice_view.extent(0); ++i) {
       std::cout << slice_view(i) << ", ";
     }
     std::cout << std::endl;
   }


The code above prints:

.. code-block:: text

   0, 1,
   4, 5,
   2, 3,
   6, 7,
