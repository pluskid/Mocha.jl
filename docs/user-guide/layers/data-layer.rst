Data Layers
~~~~~~~~~~~

.. class:: AsyncHDF5DataLayer

   Asynchronized HDF5 Data Layer. It has the same interface to :class:`HDF5DataLayer`, except that

   * The data IO is performed asynchronized with Julia coroutines. Noticeable
     speedups could typically be observed for large problems.
   * The data is read in chunks. This allows fast data shuffling of HDF5 dataset
     without using ``mmap``.

   The properties are the same as :class:`HDF5DataLayer`, with one more extra
   property controlling chunking.

   .. attribute:: chunk_size

      Default ``2^20``. The number of data points to read in each chunk. The
      data are read in chunks and cached in memory for fast random access,
      especially when data shuffling is turned on. Larger chunk size typically
      leads to better performance. Adjust this parameter according to the memory
      budget of your computing node.

      .. tip::

         * The cache only occupies host memory even when GPU backend is used for
           computation.
         * There is no correspondence between this chunk size and the *chunk
           size* property defined in a HDF5 dataset. They do not need to be the
           same.

.. class:: HDF5DataLayer


   Starting from v0.0.7, Mocha.jl contains an :class:`AsyncHDF5DataLayer`, which
   is typically more preferable than this one.

   Loads data from a list of HDF5 files and feeds them to upper layers in mini
   batches. The layer will do automatic round wrapping and report epochs after
   going over a full round of list data sources. Currently randomization is not
   supported.

   Each *dataset* in the HDF5 file should be a N-dimensional tensor. The last
   tensor dimension (the slowest changing one) is treated as the *number* dimension, and split for
   mini-batch. For more details for ND-tensor blobs used in Mocha,
   see :doc:`/dev-guide/blob`.

   The numerical types of the HDF5 datasets should either be ``Float32`` or
   ``Float64``. Even for multi-class labels, the integer class indicators should
   still be stored as floating point.

   .. note::

      For N class multi-class labels, the labels should be numerical values from
      0 to N-1, even though Julia use 1-based indexing (See
      :class:`SoftmaxLossLayer`).

   The HDF5 dataset format is compatible with Caffe. If you want to compare
   the results of Mocha to Caffe on the same data, you could use Caffe's HDF5
   Data Layer to read from the same HDF5 files Mocha is using.

   .. attribute:: source

      File name of the data source. The source should be a text file, in which
      each line specifies a file name to a HDF5 file to load.

   .. attribute:: batch_size

      The number of data samples in each mini batch.

   .. attribute:: tops

      Default ``[:data, :label]``. List of symbols, specifying the name of the
      blobs to feed to the top layers. The names also correspond to the datasets
      to load from the HDF5 files specified in the data source.

   .. attribute:: transformers

      Default ``[]``. List of data transformers. Each entry in the list should
      be a tuple of ``(name, transformer)``, where ``name`` is  a symbol of the
      corresponding output blob name, and ``transformer`` is a :doc:`data
      transformer </user-guide/data-transformer>` that should be applied to the
      blob with the given name. Multiple transformers could be given to the same
      blob, and they will be applied in the order provided here.

   .. attribute:: shuffle

      Default ``false``. When enabled, the data is randomly shuffled. Data
      shuffling is useful in training, but for testing, there is no need to do
      shuffling. Shuffled access is a little bit slower, and it requires the
      HDF5 dataset to be *mmappable*. For example, the dataset can neither be
      chunked nor be compressed. Please refer to `the documention for HDF5.jl
      <https://github.com/timholy/HDF5.jl/blob/master/doc/hdf5.md#memory-mapping>`_
      for more details.

      .. note::

         Current mmap in HDF5.jl does not work on Windows. See `issue 89 on Github
         <https://github.com/timholy/HDF5.jl/issues/89>`_.

.. class:: MemoryDataLayer

   Wrap an in-memory Julia Array as data source. Useful for testing.

   .. attribute:: tops

      Default ``[:data, :label]``. List of symbols, specifying the name of the blobs to produce.

   .. attribute:: batch_size

      The number of data samples in each mini batch.

   .. attribute:: data

      List of Julia Arrays. The count should be equal to the number of ``tops``,
      where each Array acts as the data source for each blob.

   .. attribute:: transformers

      Default ``[]``. See ``transformers`` of :class:`HDF5DataLayer`.
