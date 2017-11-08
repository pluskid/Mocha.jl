Mocha Backends
==============

A backend in Mocha is a component that carries out the actual numerical computation.
Mocha is designed to support multiple backends, and switching between different
backends should be almost transparent to the rest of the world.

There is a ``DefaultBackend`` defined which is a typealias for one of the following
backends, depending on availability. By default, ``GPUBackend`` is preferred if
CUDA is available, falling back to the ``CPUBackend`` otherwise.

Pure Julia CPU Backend
----------------------

A pure Julia CPU backend is implemented in Julia. This backend is reasonably
fast by making heavy use of Julia's built-in BLAS matrix computation library
and `performance annotations
<http://julia.readthedocs.org/en/latest/manual/performance-tips/#performance-annotations>`_
to help the LLVM-based JIT compiler produce high performance instructions.

A pure Julia CPU backend can be instantiated by calling the constructor
``CPUBackend()``. Because there is no external dependency, it should run on any
platform that runs Julia.

If you have many cores in your computer, you can play with the number of threads
used by Julia's BLAS matrix computation library by:

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

After successfully building the native extension, it can be enabled by setting
the following environment variable. In bash or zsh, execute

.. code-block:: bash

   export MOCHA_USE_NATIVE_EXT=true

before running Mocha. You can also set the environment variable inside the Julia
code:

.. code-block:: julia

   ENV["MOCHA_USE_NATIVE_EXT"] = "true"

   using Mocha

Note you need to set the environment variable **before** loading the Mocha
module. Otherwise Mocha will not load the native extension sub-module at all.

The native extension uses `OpenMP <http://openmp.org/wp/>`_ to do parallel
computation on Linux. The number of OpenMP threads used can be controlled by
the ``OMP_NUM_THREADS`` environment variable. Note that this variable is not specific
to Mocha. If you have other programs that use OpenMP, setting this environment
variable in a shell will also affect the programs started subsequently. If you
want to restrict the effect to Mocha, simply set the variable in the Julia code:

.. code-block:: julia

   ENV["OMP_NUM_THREADS"] = 1

Note that setting it to 1 disables the OpenMP parallelization. Depending on the problem
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

The native extension does not support Windows because the automatic building script
does not work on Windows. However, the native code themselve does not use any
OS specific features. If you have a compiler installed on Windows, you can try
to compile the native extension manually. However, I have **not** tested the
native extension on Windows personally.

Compile Native Extension Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native code is located in the ``deps`` directory of Mocha. Use

.. code-block:: julia

   Pkg.dir("Mocha")

to find out where Mocha is installed. You should compile it as a shared library
(DLL on Windows). However, currently the filename for the library is hard-coded
to be ``libmochaext.so``, with a ``.so`` extension, regardless of the underlying
OS.


CUDA Backend
------------

GPUs have been shown to be very effective at training large scale deep neural
networks. NVidia® recently released a GPU accelerated library of primitives for
deep neural networks called `cuDNN <https://developer.nvidia.com/cuDNN>`_. Mocha
implementes a CUDA backend by combining cuDNN, `cuBLAS
<https://developer.nvidia.com/cublas>`_ and plain CUDA kernels.

In order to use the CUDA backend, you need to have a CUDA-compatible GPU device.
The CUDA toolkit needs to be installed in order to compile the Mocha CUDA kernels.
cuBLAS is included in the CUDA distribution. But cuDNN needs to be installed
separately. You can obtain cuDNN from `Nvidia's website
<https://developer.nvidia.com/cuDNN>`_ by registering as a CUDA developer for
free.

.. note::

   Mocha is tested on CUDA 8.0 and cuDNN 5.1 on Linux. Since cuDNN typically do not
   keep backward compatibility in the APIs, it is not guaranteed to run on
   different versions.

Before using the CUDA backend, the Mocha kernels needs to be compiled. The kernels
are located in ``src/cuda/kernels``. Please use ``Pkg.dir("Mocha")`` to find out
where Mocha is installed on your system. We have included a Makefile for
convenience, but if you don't have ``make`` installed, the command for compiling is
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

Note that instead of instantiating a ``CPUBackend``, you now construct
a ``GPUBackend``. The environment variable needs to be set **before** loading
Mocha. It is designed to use conditional loading so that the pure CPU backend
can still run on machines which don't have a GPU device or don't have the CUDA
library installed. If you have multiple GPU devices on one node, the environment
variable ``MOCHA_CUDA_DEVICE`` can be used to specify the device ID to use. The
default device ID is ``0``.

Recompiling Kernels
~~~~~~~~~~~~~~~~~~~

When you upgrade Mocha to a higher version, the source code for some CUDA kernel
implementations might have changed. Mocha will compile the timestamps for the
compiled kernel and the source files. An error will be raised if the compiled kernel
file is found to be older than the kernel source files. Simply following the procedures
above to compile the kernel again will solve this problem.

