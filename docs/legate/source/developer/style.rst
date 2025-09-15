..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

==========
Code Style
==========

The legate source code "style" is heavily based on the `Google style guide
<https://google.github.io/styleguide/>`_ and is rigorously enforced through the use of
``clang-format`` and other automatic linting tools.

While seemingly trivial at first, a unified code style helps to significantly increase the
readability of a code-base. If all code is written in a similar style and flow, then a
reader may begin to form implicit assumptions (which should hold), and does not need to
adapt to new forms. In short, the point of a common style is not to enforce one that is
"pretty" or "elegant". It is first and foremost to improve the understand-ability of the
code.

However, there are some areas where automatic enforcement is not yet sufficient, or not
yet possible. These are discussed below. In all other cases the following rules apply:

#. **Some particular format is mandated by automated style-checkers.**

   Unless there are extenuating circumstances (such as a bug in the style-checker, or
   extreme loss in readability), **the word of the checker is law**. You should **not**
   attempt to override the checker via e.g. ``// clang-format off`` unless it has been
   specifically approved by a maintainer.

#. **No particular format is mandated by automated style-checkers.**

   The author should try to **stay consistent with existing practice**. More often than
   not, a similar construct exists somewhere else in the library, which the author should
   apply to their own situation. The point of this rule is to make future refactoring
   easier to do. Instead of needing to special-case future checkers for every bespoke
   solution (and potentially miss cases), they can be tailored to just a handful of cases.

------------------------------

Classes
-------

#. 1 class definition per header file.

#. Try to avoid nested classes that are visible outside the class. Nested classes cannot
   be forward-declared and therefore require including the entire header file of the
   enclosing class.

#. Member functions (or constructors) taking a resource which they intend to own should
   take the resource by value and ``std::move()`` it into place. This avoids a double copy
   in case the argument is constructed in-place at the call-site, and is no less efficient
   if a copy is required.

   .. code-block:: cpp


      // GOOD, the object intends to own f, so it should take by value
      void MyObject::add_foo(Foo f)
      {
        foos_.emplace_back(std::move(f));
      }


      // BAD
      void MyObject::add_foo(const Foo& f)
      {
        foos_.emplace_back(f);
      }

      // Bad because this now causes a copy. The in-place constructed Foo is copied into
      // the member variable instead of being moved.
      my_object.add_foo(Foo{...});

#. Do not hold raw pointers to resources inside a class unless absolutely
   necessary. Prefer to hold smart pointers or ``std::reference_wrapper``. If this is not
   possible, the initializer for the pointer must be done via reference.

   .. code-block:: cpp

      // GOOD
      class Foo {
      public:
        // Good, take b by reference by converting to pointer internally.
        Foo(const Bar &b) : bar_{&b} { }

        // Good, return a reference to bar_, not the pointer.
        const Bar &bar() const { return *bar_; }

      private:
        const Bar *bar_{};
      };

      // BAD
      class Foo {
      public:
        // Bad, should take a reference to indicate to the caller "this resource must exist"
        // and allow compiler to check this for us.
        Foo(const Bar *b) : bar_{b} { }

        // Bad, should return a reference to bar_, not the pointer.
        const Bar *bar() const { return bar_; }

      private:
        const Bar *bar_{};
      };


#. Do not befriend other classes. If another class must have private access to specific
   member variables or member functions that should otherwise not be exposed, expose these
   functions as public methods with private "keys":

   .. code-block:: cpp

      class Foo;

      class Bar {
      public:

      class UseResourceKey {
        // Note, default constructor is explicitly made private. No-one except for classes
        // or functions made friends may construct this "key" now.
        UseResourceKey() = default;

        // OK, allow any function of Foo to call the private function.
        friend class Foo;

        // Best, allow only specific functions of Foo to call the private function.
        friend void Foo::use_resource(...);

        // Usually you will want the current class as a friend so that it may call the
        // function as well.
        friend class Bar;
      };

      // Function must be public, so that Foo is allowed to call it. But it is for all
      // intents and purposes private, because no-one except for specific classes or
      // functions can construct the first "key" argument.
      void use_resource(UseResourceKey, ...);

      private:
        Widget resource_{};
      };

#. Do not expose your members directly. Instead, make them ``private`` and provide
   ``public`` accessor functions to them. This serves to encapsulate the resource and
   provide clear ownership semantics.

   This also aides in debugging. When tracking when a member's variables are changed, it
   is much easier to place a breakpoint on a function than it is to place a watchpoint on
   a particular member variable's address (which may change due to serdez).

#. Do not expose mutable references to members unless absolutely necessary. Instead,
   provide functions that perform the mutation directly. Not only should a class own its
   resources, it should have complete control of any modifications of those resources.

Functions
---------

.. _label_short_and_sweet_functions:

#. Functions should be short and sweet. They should do one thing, and do it very well. A
   general rule is that if a particular function exceeds 50 lines, or 3 levels of nesting
   then it should be broken up into smaller pieces.

#. Functions that are used only in the current translation unit should be moved inside
   anonymous namespaces.

#. Functions should not take in-out parameters if this can be avoided. If possible, if a
   function produces some effect, it should return it as a value.

#. Do not use raw pointer arguments for input parameters. Use either smart pointers,
   references, or, when necessary, ``std::reference_wrapper``. Similarly, never use
   ``nullptr`` to indicate lack of value for a pointer-like type. If a pointer argument is
   optional to the caller, then wrap it in a ``std::optional`` instead. Legate assumes all
   pointers to be non-NULL.

#. Functions taking out-parameters (if it cannot be avoided), should take them as a
   pointer, not by reference. Out-parameters must come last in the function definition. For example:

   .. code-block:: cpp

      // GOOD
      void foo(const T& in, T value, T* out_1, T* out_2);

      // BAD: out-param taken by reference
      void foo(T& out);

      // BAD: out-param must come last
      void foo(T* out, const T& in);

      // BAD: all out-params must come last
      void foo(const T& in, T* out, T value, T* out_2);


   This is to help readability at the call-site:

   .. code-block:: cpp

      SomeType f;

      foo(f); // Does foo() modify f? Who knows?
      foo(&f); // OK, foo() potentially modifies f


#. Functions that only need to take a view of linear memory or containers (e.g. those
   taking a ``const std::vector&``), should always take that view as a
   ``legate::Span<const T>``. For example:

   .. code-block:: cpp

      // GOOD
      void foo(Span<const int> values);

      // BAD
      void foo(const std::vector<int>& values);


   Likewise, functions that take a ``const std::string&`` should instead take a
   ``std::string_view``. Both ``Span`` and ``std::string_view`` should always be taken by
   value.

#. For functions in header files, declarations and definitions must always be separate. If
   the function is a template (or otherwise very very small), the definition should go in
   the corresponding ``.inl``. If not, the definition should go in the ``.cc``. This also
   applies for any member functions of classes.

   This does not apply to translation unit-local functions defined in anonymous
   namespaces. These may be defined and declared in the same place.

   .. note::

      "Small" in this case refers to the code-gen, not the size of the source code
      itself. Anything that is reasonably expected to be completely optimized away is
      considered "small". This is usually either:

      - Constructors, where everything is ``std::move()``-ed (which usually end up
        compiling away to a bunch of pointer shuffling).
      - Getters returning some kind of reference (which end up compiling away to just
        returning a pointer).

      Things that are *not* considered "small" are:

      - Functions that throw exceptions.
      - Functions that allocate memory.
      - Functions that call other non-small functions.

   To elaborate on the rationale for this rule: the goal of defining things in an ``.inl``
   is to facilitate compiler optimizations. In the snippet below:

   .. code-block:: cpp

      struct Foo {
        int get_bar() const { return this->bar_; }

        int bar_;
      };

      int foo(const Foo& f)
      {
        return f->get_bar();
      }


   The compiler should be able to see ``get_bar()``, because with it, it sees that the
   code effectively reduces to:

   .. code-block:: cpp

      int foo(const Foo& f)
      {
        return f.bar_;
      }

   If, however, after unwrapping the various functions, the compiler still has to emit an
   indirect function call (to a function defined in the ``.cc``), then there is no point
   in having it be in the ``.inl``.

   In this case, it is better to have it defined in the ``.cc``, where the compiler can
   inline the other function call, resulting in more efficient code overall.

#. All functions, public or private, must be documented with a proper doxygen
   docstring. This includes translation-unit local functions, such as those in anonymous
   namespaces.

   If the public and private variants are more or less identical in terms of arguments
   (i.e. the public is a pass-through for the private), then the private docstring does
   not need to repeat the docstring of the public. Instead, it should include text that
   references the public variant (e.g. "see legate::Foo::bar() for further discussion"),
   and document only the "private" effects of calling the function (e.g. "this function
   modifies <some internal property> to state X and therefore shouldn't be called before
   XYZ").

   We could have adopted a rule that "small" or "obvious" functions shouldn't get this
   treatment, but then there would be endless bike-shedding on whether a particular
   function is small or obvious enough. Combined with the fact that it is very easy to
   hide code in operators in C++, it is easier to mandate that all functions be
   documented.


Variables
---------

#. Declarations and code should be separated by a single empty line. Separating
   declarations and logic helps to improve readability of the code. For example:

   .. code-block:: cpp


      // GOOD
      std::size_t SIZE = 10;
      std::vector<int> y;

      y.reserve(SIZE);

      std::size_t vec_size = y.size();

      // BAD: no empty lines before or after declarations
      std::size_t SIZE = 10;
      std::vector<int> y;
      y.reserve(SIZE);
      std::size_t vec_size = y.size();

#. Use ``auto`` whenever you already name a type. For example, when using
   ``static_cast()`` (or the other casts), or when initializing variables. Additionally,
   use ``auto`` when the resulting type would be a large template or other such construct
   whose type may detract from the readability of the code. For example:

   .. code-block:: cpp

      // GOOD
      auto* x = static_cast<int*>(y);
      auto object = get_complex_type_object();
      auto sv = std::string_view{};

      // BAD: we already know it will be an int* from the cast
      int* x = static_cast<int*>(x);

      // BAD: the type of the variable both is very complex, and needlessly pollutes the code
      std::unordered_map<std::unique_ptr<SomeType, CustomDeleter>, std::deque<Foo, CustomDeleter>> object
        = get_complex_type_object();

      // BAD: we already know what type it is based on the constructor name
      std::string_view sv = std::string_view{}

#. When using ``auto`` with pointers, they should be matched with ``auto*``.

#. When using ``auto`` with references, they should be matched with ``auto&``.

#. If it is unclear whether the return value of a function returns by ``const`` reference,
   value, or r-value reference, the type should be matched with ``auto&&``.

#. Declare variables as ``const`` whenever possible.

#. Variables should be named readably. Unfortunately, there is no hard and fast rule to
   follow for this, and generally speaking it relies on a programmer's good judgment,
   however a few hard rules are:

   #. Never use Hungarian typing. The compiler knows what type it is. You also know what
      type it is, because the function is short enough to fit on your screen (as per the
      short functions :ref:`rule <label_short_and_sweet_functions>`). There is no need to
      encode this in the name of the variable as well.
   #. Never use acronyms. You might know what "``dcv``" (dimension-less color vector)
      means now, but nobody else does, and neither will you in 4 week's time.
   #. Use commonly understood names for common things. For example, ``for``-loop indices
      should almost always be ``i``, ``j``, or ``k``. If something returns an iterator, it
      should probably be called ``it``, not e.g. ``pos``, ``idx``, or ``finder``.
   #. Don't use verbose names. ``iterator_into_vector`` is no more informative than
      e.g. ``it``. ``temporary_vector`` is no better (arguably, it is worse) than just ``tmp``.

#. Instead of

   .. code-block:: cpp

      Type var;

      if (cond) {
        var = ...;
      } else {
        var = ...;
      }

   Prefer returning a value from an immediate lambda:

   .. code-block:: cpp

      const auto var = [&] {
        if (cond) {
          return ...;
        }
        return ...;
      }();  // Note, lambda is executed immediately

   This has 2 main benefits:

    #. It ensures that ``var`` is always initialized (you'll never forget to set it if you
       have a lot of if branches).
    #. You can make ``var`` ``const``.

#. Do not use pointers to indicate lack of value. If a resource may or may not exist, then
   use ``std::optional<T>`` instead. This also goes for smart pointers; as described
   above, Legate assumes all pointers (smart or otherwise) are non-null and point to a
   value.

   The rationale for this is that compiler will warn about unchecked accesses to
   ``std::optional<T>`` (which is undefined behavior if the optional does not contain a
   value), while they won't complain about NULL-pointer dereferences. ``std::optional`` is
   also more explicit and self-documenting. It tells the reader unambiguously that "this
   might not exist".

Misc
----

#. Use of ``LEGATE_CHECK()`` and ``LEGATE_ABORT()``:

   ``LEGATE_CHECK()`` and ``LEGATE_ABORT()`` will abort the program if the operand
   evaluates to false. As such they are to be used only when catching *library*
   mistakes. They are semantically equivalent to the standard ``abort()`` macro, in that
   they enforce foundational pre-conditions or post-conditions which must never be
   violated in bug-free library code.

   For instance, they must never be used to check user-supplied values. In these cases,
   legate should throw an exception that can be caught and handled by the user. For
   example:

   .. code-block:: cpp


      void foo(int dim)
      {
        if (dim < 0) {
          // The user has given us a bad value, so we should handle this by exception.
          throw an_exception{...};
        }

        dim = detail::fixup_dim(dim);
        // The internal function returned a non-positive dim. This can only happen if
        // there exists a bug within legate itself, in which case the library is probably
        // in an inconsistent state and the program cannot continue.
        LEGATE_CHECK(dim > 0);
      }

#. Inline literal arguments (such as literal numbers or ``true``/ ``false`` constants)
   should always have an inline comment with the argument name they refer to, except when
   the intent is "obvious". Of course "what is obvious" is a whole other exercise in
   bike-shedding of its own, but when in doubt err on the side of caution:

   .. code-block:: cpp

      // GOOD: obvious
      foo.set_dim(4);
      foo.is_cached(true);

      // GOOD: labeled arguments
      foo.create_bar(..., /* dim */ 5);
      foo.to_string(/* provenance */ true);

      // BAD: not obvious what the arguments refer to
      foo.create_bar(..., 5);
      foo.to_string(true);
