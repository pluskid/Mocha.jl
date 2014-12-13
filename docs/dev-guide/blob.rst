Blob
====

Blob is the fundamental data representation in Mocha. It is used as both data
(e.g. mini-batch of data samples) and parameters (e.g. filters of a convolution
layer). Conceptually, a blob is a 4D tensor. Following the vision (and Caffe)
convention, the four dimensions are called *width*, *height*, *channels* and
*num*. The fastest changing dimension is *width* and slowest changing dimension
is *num*.

.. note::

   The memory layout of a blob in Mocha is compatible with Caffe's blob. So
   a blob (e.g. layer parameters) in Mocha can be saved to HDF5 and load it from
   Caffe without doing any dimension permutation, and vise versa. However, since
   Julia use column-major convention for tensor and matrix, and Caffe use
   row-major convention, in Mocha API, the order of the four dimensions is
   width, height, channels, and num, while in Caffe API, it is num, channels,
   height, width.

Each backend has its own blob implementation, as a subtype of :class:`Blob`. For
example, a blob in the CPU backend is a shallow wrapper of a Julia ``Array``
object, while a blob in the GPU backend references to a piece of GPU memory.

Constructors and Destructors
----------------------------

A backend-dependent blob can be created with the following function:

.. function::
   make_blob(backend, data_type, dims)

   ``dims`` is a ``NTuple{4, Int}``, specifying the four dimensions of the blob
   to be created. Currently ``data_type`` should be either ``Float32`` or
   ``Float64``.

Several helper functions are also provided:

.. function:: make_blob(backend, data_type, width, height, channels, num)

   Spell out the four dimensions explicitly.

.. function:: make_blob(backend, array)

   ``array`` is a Julia ``AbstractArray``. This makes a blob with the same data
   type and shape as ``array`` and initialize the blob contents with ``array``.

.. function:: make_zero_blob(backend, data_type, dims)

   Create a blob and initialize with zeros.

.. function:: reshape_blob(backend, blob, new_dims)

   Create a reference to an existing blob with a possiblely different shape.
   The behavior is the same as Julia's ``reshape`` function on an array: the new
   blob shares data with the existing one.

The resources of a blob could be released by calling

.. function:: destroy(blob)

Note the resources need to be released explicitly. A Julia blob object being
GC-ed does not release the underlying resource automatically.

Accessing Properties of a Blob
------------------------------

The blob implements some simple API for a Julia array:

.. function:: eltype(blob)

   Get the element type of the blob.

.. function:: size(blob)

   Get the shape of the blob. The return value is a ``NTuple{4, Int}``.

.. function:: size(blob, dim)

   Get the size at a particular dimension. For example, ``size(blob, 1)`` gets
   the width of a blob.

.. function:: length(blob)

   Get the total number of elements in a blob.

The wrappers ``get_width``, ``get_height``, ``get_chann`` and ``get_num`` could
also be used.

Accessing Data of a Blob
------------------------

Because accessing GPU memory is costly, a blob does not has interface to do
element-wise accessing. The data could either be manipulated in
a backend-dependent manner, relying on the underlying implementation details; or
in a backend-independent way by copying the contents back and to a Julia array.

.. function:: copy!(dst, src)

   Copy the contents of ``src`` to ``dst``. ``src`` and ``dst`` could be either
   a blob or a Julia array.

The following utilities could be used to initialize the contents of a blob

.. function:: fill!(blob, value)

   Fill every element of ``blob`` with ``value``.

.. function:: erase!(blob)

   Fill ``blob`` with zeros. Depending on the implementation, ``erase!(blob)``
   might be more efficient than ``fill!(blob, 0)``.
