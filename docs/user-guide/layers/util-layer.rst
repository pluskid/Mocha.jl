Utility Layers
~~~~~~~~~~~~~~

.. class:: ConcatLayer

   Concatenates multiple blobs into a single blob along the specified dimension. Except in
   the concatenation dimension, the shapes of the blobs being concatenated have to
   be the same.

   .. attribute:: dim

      Default 3 (channel). The dimension to concatenate.

   .. attribute:: bottoms

      Names of the blobs to be concatenated.

   .. attribute:: tops

      Name of the concatenated output blob.

.. class:: MemoryOutputLayer

   Takes some blobs in the network and collect their data during forward pass of
   the network as a list of julia ``Array`` objects. Useful when doing in-memory
   testing for collecting the output. After running the forward pass of the
   network, the ``outputs`` field of the corresponding layer state object will
   contain a vector of the same size as the ``bottoms`` attribute. Each element
   of the vector is a list of tensors (julia ``Array`` objects), each tensor
   corresponds to the output in a mini-batch.

   .. attribute:: bottoms

      A list of names of the blobs in the network to store.

.. class:: HDF5OutputLayer

   Takes some blobs in the network and writes them to a HDF5 file.
   Note that the target HDF5 file will be overwritten when the network is first
   constructed, but later iterations will **append** data for each mini-batch.
   This is useful for storing the final predictions or the intermediate
   representations (feature extraction) of a network.

   .. attribute:: filename

      The path to the target HDF5 file.

   .. attribute:: force_overwrite

      Default ``false``. When the layer tries to create the target HDF5 file, if
      this attribute is enabled, it will overwrite any existing file (with
      a warning printed). Otherwise, it will raise an exception and refuse to
      overwrite the existing file.

   .. attribute:: bottoms

      A list of names of the blobs in the network to store.

   .. attribute:: datasets

      Default ``[]``. Should either be empty or a list of ``Symbol`` of the same length as
      ``bottoms``. Each blob will be stored as an HDF5 dataset in the target HDF5
      file. If this attribute is given, the corresponding symbol in this list is
      used as the dataset name instead of the original blob's name.

.. class:: IdentityLayer

   An Identity layer maps inputs to outputs without changing anything. This can
   be useful as a glue layer to rename some blobs. There is no data-copying for
   this layer.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same.

.. class:: Index2OnehotLayer

   A utility layer that could convert category class into one-hot encoded
   vector. For example, for K classes, input j is converted into a vector of
   size K, with all zeros, but the (j-1)-th entry 1.

   .. attribute:: dim

      The dimension to operate on. The input must have size 1 on this dimension,
      i.e. ``size(input, dim) == 1``. And the value should be integers from 0 to
      (K-1).

   .. attribute:: n_class

      Number of categories, i.e. K as described above.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same. But they will be operated on
      the same dimension, and the ``n_class`` for them are the same.

.. class:: ReshapeLayer

   Reshapes a blob. Can be useful if, for example, you want to make the *flat*
   output from an :class:`InnerProductLayer` *meaningful* by assigning each
   dimension spatial information.

   Internally, no data is copied. The total number of elements in
   the blob tensor after reshaping has to be the same as the original blob
   tensor.

   .. attribute:: shape

      Has to be an ``NTuple`` of ``Int`` specifying the new shape. Note that the new
      shape does not include the last (mini-batch) dimension of a data blob. So
      a reshape layer cannot change the mini-batch size of a data blob.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same. But the feature dimensions
      (i.e. the product of the first 3 dimensions) have to be the same.

.. class:: SplitLayer

   A Split layer produces identical copies of the input. The number of copies
   is determined by the length of the ``tops`` property. During back propagation,
   derivatives from all the output copies are added together and propagated down.

   This layer is typically used as a helper to implement some more complicated
   layers.

   .. attribute:: bottoms

      Input blob names, only one input blob is allowed.

   .. attribute:: tops

      Output blob names, should be more than one output blobs.

   .. attribute:: no_copy

      Default ``false``. When ``true``, no data is copied in the forward pass.
      In this case, all the output blobs share data. When, for example, an
      *in-place* layer is used to modify one of the output blobs, all the other
      output blobs will also change.

