Mocha Backends
==============

A backend in Mocha is a component that carries out actual numerical computation.
Mocha is designed to support multiple backends, and switching between different
backends should be almost transparent to the rest of the world.

Pure Julia CPU Backend
----------------------

A pure Julia CPU backend is implemented in Julia. This backend is reasonably
fast by making heavy use of the Julia's built-in BLAS matrix computation library
and `performance annotations
<http://julia.readthedocs.org/en/latest/manual/performance-tips/#performance-annotations>`_
to help the LLVM-based JIT compiler producing high performance instructions.

A pure Julia CPU backend could be instantiated by calling the constructor
``CPUBackend()``. Because there is no external dependency, it should runs on any
platform that runs Julia.

If you have many cores in your computer, you can play with the number of threads
used by the Julia's BLAS matrix computation library by:

.. code-block:: julia

   blas_set_num_threads(N)

Depending on the problem size and a lot of other factors, using larger N is
not necessarily faster.

CPU Backend with Native Extension
---------------------------------

Mocha comes with C++ implementations of some bottleneck computations for the CPU
backend. In order to use the native extension, you need to build the native code
first (if it is not built automatically when installing the package).

.. code-block:: julia

   Pkg.build("Mocha")

After successfully building the native extension, it could be enabled by setting
the environment variable. On bash or zsh, execute

.. code-block:: bash

   export MOCHA_USE_NATIVE_EXT=true

before running Mocha. You can also set the environment variable inside the Julia
code:

.. code-block:: julia

   ENV["MOCHA_USE_NATIVE_EXT"] = "true"

   using Mocha

Note you should set the environment variable **before** loading the Mocha
module. Otherwise Mocha will not load the native extension sub-module at all.

The native extension uses `OpenMP <http://openmp.org/wp/>`_ to do parallel
computation on Linux. The number of OpenMP threads used could be controlled by
the ``OMP_NUM_THREADS`` environment variable. Note this variable is not specific
to Mocha. If you have other programs that uses OpenMP, setting this environment
variable in a shell will also affect those problems started subsequently. If you
want to restrict to Mocha, simply set the variable in the Julia code:

.. code-block:: julia

   ENV["OMP_NUM_THREADS"] = 1

Note setting to 1 disabled the OpenMP parallelization. Depending on the problem
size and a lot of other factors, using multi-thread OpenMP parallelization is
not necessarily faster because of the overhead of multi-threads.

The parameter for the number of threads used by the BLAS library applies to the
CPU backend with native extension, too.

OpenMP on Mac OS X
~~~~~~~~~~~~~~~~~~

When compiling the native extension on Mac OS X, you will get a warning that
OpenMP is disabled. This is because currently clang, the built-in compiler for
OS X, does not officially support OpenMP yet. If you want to try OpenMP on OS X,
`Clang-OMP <http://clang-omp.github.io/>`_

Native Extension on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The automatic building script does not work on Windows.
