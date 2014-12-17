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
please refer to `Clang-OMP <http://clang-omp.github.io/>`_ and compile manually
(see below).

Native Extension on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native extension does not support Windows because automatic building script
does not work on Windows. However, the native codes themselves does not use any
OS specific features. If you have a compiler installed on Windows, you could try
to compile the native extension manually. However, I have **not** tested the
native extension on Windows personally.

Compile Native Extension Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native codes are located in the ``deps`` directory of Mocha. Use

.. code-block:: julia

   Pkg.dir("Mocha")

to find out where Mocha is installed. You should compile it as a shared library
(DLL on Windows). However, currently the filename for the library is hard-coded
to be ``libmochaext.so``, with a ``.so`` extension, regardless of the underlying
OS.


CUDA Backend
------------

GPU has been shown to be very effective at training large scale deep neural
networks. NVidia® recently released a GPU accelerated library of primitives for
deep neural networks called `cuDNN <https://developer.nvidia.com/cuDNN>`_. Mocha
implemented a CUDA backend by combining cuDNN, `cuBLAS
<https://developer.nvidia.com/cublas>`_ and plain CUDA kernels.

In order to use the CUDA backend, you need to have CUDA-compatible GPU devices.
The CUDA toolkit should be installed in order to compile the Mocha CUDA kernels.
cuBLAS is included in CUDA distribution. But cuDNN needs to be installed
separately. You could obtain cuDNN from `Nvidia's website
<https://developer.nvidia.com/cuDNN>`_ by registering as a CUDA developer for
free.

.. note::

   * cuDNN requires CUDA 6.5 to run.
   * Mocha v0.0.1 ~ v0.0.4 use cuDNN 6.5 R1, which is only available on Linux
     and Windows.
   * Mocha v0.0.5 and higher uses cuDNN 6.5 R2, which is also
     available on Mac OS X.
   * cuDNN 6.5 R2 is **not** backward compatible with cuDNN 6.5 R1.

Before using the CUDA backend, Mocha kernels needs to be compiled. The kernels
are located in ``src/cuda/kernels``. Please use ``Pkg.dir("Mocha")`` to find out
where Mocha is installed on your system. We have included a Makefile for
convenience, but if you don't have ``make`` installed, the compiling command is
as simple as

.. code-block:: bash

   nvcc -ptx kernels.cu

After compiling the kernels, you can now start to use the CUDA backend by
setting the environment variable ``MOCHA_USE_CUDA``. For example:

.. code-block:: julia

   ENV["MOCHA_USE_CUDA"] = "true"

   using Mocha

   backend = GPUBackend()
   init(backend)

   # ...

   shutdown(backend)

Note instead of instantiate a ``CPUBackend``, you now construct
a ``GPUBackend``. The environment variable should be set **before** loading
Mocha. It is designed to use conditional loading so that the pure CPU backend
could still run on machines without any GPU device or CUDA library installed.

Recompiling Kernels
~~~~~~~~~~~~~~~~~~~

When you upgrade Mocha to a higher version, the source code for some CUDA kernel
implementations might be changed. Mocha will compile the timestamps for the
compiled kernel and the source files. An error will raise if the compiled kernel
file is found older than the kernel source files. Just follow the procedures
above to compile the kernel again will solve this problem.

