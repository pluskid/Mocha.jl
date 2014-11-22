Data Transformers
=================

Data transformers apply transformations to data. Note the transformations are
limited to simple, in-place operations that does not change the shape of the
data. If more complicated transformations like random projection or feature
mapping are needed, consider using a data transformation **layer** instead.

.. class:: DataTransformers.SubMean

   Subtract mean from the data. The transformer does not have enough information
   to compute the data mean, thus the mean should be computed in advance.

   .. attribute:: mean_blob

      Default ``NullBlob()``. A blob containing the mean data.

   .. attribute:: mean_file

      Default "". When ``mean_blob`` is a ``NullBlob``, this could be used to
      specify a HDF5 file containing the mean. The mean should be stored with
      the name ``mean`` in the HDF5 file.

.. class:: DataTransformers.Scale

   Do elementwise scaling for the data. This is useful in the cases, for
   example, when you want to scale the data into, say the range [0,1].

   .. attribute:: scale

      Default 1.0. The scale to apply.
