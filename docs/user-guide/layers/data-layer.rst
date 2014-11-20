Data Layers
~~~~~~~~~~~

.. class:: HDF5DataLayer

   Load data from a list of HDF5 files and feed them to upper layers in mini
   batches. The layer will do automatic round wrapping and report epochs after
   going over a full round of list data sources. Currently randomization is not
   supported.

   Each *dataset* in the HDF5 file should be a 4D tensor. Using the naming
   convention for image datasets, the four dimensions are (width, height,
   channels, number). Here the fastest changing dimension is *width*, while the
   slowest changing dimension is *number*. Mini-batch splitting will occur in
   the *number* dimension. For more details for 4D tensor blobs used in Mocha,
   see :doc:`/dev-guide/blob`.

   Currently, the dataset should be explicitly in 4D tensor format. For example,
   if the label for each sample is only one number, the HDF5 dataset should
   still be created with dimension (1, 1, 1, number).

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

.. class:: MemoryDataLayer

   Wrap an in-memory Julia Array as data source. Useful for testing.

   .. attribute:: tops

      List of symbols, specifying the name of the blobs to produce.

   .. attribute:: batch_size

      The number of data samples in each mini batch.

   .. attribute:: data

      List of Julia Arrays. The count should be equal to the number of ``tops``,
      where each Array acts as the data source for each blob.
