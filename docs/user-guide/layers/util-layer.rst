Utility Layers
~~~~~~~~~~~~~~

.. class:: ConcatLayer

   Concating multiple blobs into one along the specified dimension. Except in
   the concatenation dimension, the shapes of the blobs being concatenated should
   be the same.

   .. attribute:: dim

      Default 3 (channel). The dimension to concat.

   .. attribute:: bottoms

      Names of the blobs to be concatenated.

   .. attribute:: tops

      Name of the concatenated output blob.

.. class:: HDF5OutputLayer

   Take some blobs in the network and write the blob contents to a HDF5 file.
   Note the target HDF5 file will be overwritten when the network is first
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

   Identity layer maps inputs to outputs without changing anything. This could
   be useful as glue layers to rename some blobs. There is no data-copying for
   this layer.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer could take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same.

.. class:: ReshapeLayer

   Reshape a blob. Can be useful if, for example, you want to make the *flat*
   output from an :class:`InnerProductLayer` *meaningful* by assigning each
   dimension spatial information.

   Internally there is no data copying going on. The total number of elements in
   the blob tensor after reshaping should be the same as the original blob
   tensor.

   .. attribute:: shape

      Should be an ``NTuple`` of ``Int`` specifying the new shape. Note the new
      shape does not include the last (mini-batch) dimension of a data blob. So
      a reshape layer cannot change the mini-batch size of a data blob.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer could take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same. But the feature dimensions
      (product of the first 3 dimensions) should be the same.

.. class:: SplitLayer

   Split layer produces identical copies of the input. The number of copies
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

