Data Layers
~~~~~~~~~~~

.. class:: HDF5DataLayer

   Load data from a list of HDF5 files and feed them to upper layers in mini
   batches. The layer will do automatic round wrapping and report epochs after
   going over a full round of list data sources. Currently randomization is not
   supported.

   .. attribute:: source

      File name of the data source. The source should be a text file, in which
      each line specifies a file name to a HDF5 file to load.

   .. attribute:: batch_size

      The number of data samples in each mini batch.

   .. attribute:: tops

      List of symbols, specifying the name of the blobs to feed to the top
      layers. The names also correspond to the datasets to load from the HDF5
      files specified in the data source.

.. class:: MemoryDataLayer

   Wrap an in-memory Julia Array as data source. Useful for testing.

   .. attribute:: tops

      List of symbols, specifying the name of the blobs to produce.

   .. attribute:: batch_size

      The number of data samples in each mini batch.

   .. attribute:: data

      List of Julia Arrays. The count should be equal to the number of ``tops``,
      where each Array acts as the data source for each blob.
