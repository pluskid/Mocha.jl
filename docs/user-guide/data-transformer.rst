Data Transformers
=================

Data transformers apply transformations to data. Note that the transformations are
limited to simple, in-place operations that do not change the shape of the
data. If more complicated transformations like random projection or feature
mapping are needed, consider using a data transformation **layer** instead.

.. class:: DataTransformers.SubMean

   Subtract mean from the data. The transformer does not have enough information
   to compute the data mean, thus the mean should be computed in advance.

   .. attribute:: mean_blob

      Default ``NullBlob()``. A blob containing the mean.

   .. attribute:: mean_file

      Default ``""``. When ``mean_blob`` is a ``NullBlob``, this can be used to
      specify a HDF5 file containing the mean. The mean should be stored with
      the name ``mean`` in the HDF5 file.

.. class:: DataTransformers.Scale

   Perform elementwise scaling of the data. This is useful, for
   example, when you want to scale the data to, say, the range [0,1].

   .. attribute:: scale

      Default 1.0. The scaling factor to apply.
