Blob
====

A blob is the fundamental data representation in Mocha. It is used for both data
(e.g. mini-batch of data samples) and parameters (e.g. filters of a convolution
layer). Conceptually, a blob is an N-dimensional tensor.

For example, in vision, a data blob is usually a 4D-tensor. Following the vision
(and Caffe) convention, the four dimensions are called *width*, *height*,
*channels* and *num*. The fastest changing dimension is *width* and slowest
changing dimension is *num*.

.. note::

   The memory layout of a blob in Mocha is compatible with Caffe's blob. So
   a blob (e.g. layer parameters) in Mocha can be saved to HDF5 and loaded from
   Caffe without doing any dimension permutation, and vise versa. However, since
   Julia uses the column-major convention for tensor and matrix data, and Caffe uses
   the row-major convention, in Mocha API, the order of the four dimensions is
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

   ``dims`` is an ``NTuple``, specifying the dimensions of the blob to be
   created. Currently ``data_type`` should be either ``Float32`` or ``Float64``.

Several helper functions are also provided:

.. function:: make_blob(backend, data_type, dims...)

   Spell out the dimensions explicitly.

.. function:: make_blob(backend, array)

   ``array`` is a Julia ``AbstractArray``. This creates a blob with the same data
   type and shape as ``array`` and initializes the blob contents with ``array``.

.. function:: make_zero_blob(backend, data_type, dims)

   Create a blob and initialize it with zeros.

.. function:: reshape_blob(backend, blob, new_dims)

   Create a reference to an existing blob with a possiblely different shape.
   The behavior is the same as Julia's ``reshape`` function on an array: the new
   blob shares data with the existing one.


.. function:: destroy(blob)

    Release the resources of a blob.

.. note::

    The resources need to be released explicitly. A Julia blob object being
    GC-ed does not release the underlying resource automatically.

Accessing Properties of a Blob
------------------------------

The blob implements a simple API similar to a Julia array:

.. function:: eltype(blob)

   Get the element type of the blob.

.. function:: ndims(blob)

   Get the tensor dimension of the blob. The same as ``length(size(blob))``.

.. function:: size(blob)

   Get the shape of the blob. The return value is an ``NTuple``.

.. function:: size(blob, dim)

   Get the size along a particular dimension. ``dim`` can be negative. For
   example, ``size(blob, -1)`` is the same as ``size(blob)[end]``. For
   convenience, if ``dim`` exceeds ``ndims(blob)``, the function returns ``1``
   instead of raising an error.

.. function:: length(blob)

   Get the total number of elements in a blob.

.. function:: get_width(blob)

   The same as ``size(blob, 1)``.

.. function:: get_height(blob)

   The same as ``size(blob, 2)``.

.. function:: get_num(blob)

   The same as ``size(blob, -1)``.

.. function:: get_fea_size(blob)

   The the *feature size* in a blob, which is the same as
   ``prod(size(blob)[1:end-1])``.

The wrapper ``get_chann`` was removed when Mocha upgraded from
4D-tensors to general ND-tensors, because the channel dimension is usually
ambiguous for general ND-tensors.

Accessing Data of a Blob
------------------------

Because accessing GPU memory is costly, a blob does not have an interface to do
element-wise accessing. The data can be either manipulated in
a backend-dependent manner, relying on the underlying implementation details, or
in a backend-independent way by copying the contents from and to a Julia array.

.. function:: copy!(dst, src)

   Copy the contents of ``src`` to ``dst``. ``src`` and ``dst`` can be either
   a blob or a Julia array.

The following utilities can be used to initialize the contents of a blob

.. function:: fill!(blob, value)

   Fill every element of ``blob`` with ``value``.

.. function:: erase!(blob)

   Fill ``blob`` with zeros. Depending on the implementation, ``erase!(blob)``
   might be more efficient than ``fill!(blob, 0)``.
