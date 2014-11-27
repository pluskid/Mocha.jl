Utility Layers
~~~~~~~~~~~~~~

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
