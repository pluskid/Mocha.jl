Statistics Layers
~~~~~~~~~~~~~~~~~

.. class:: AccuracyLayer

   Compute and accumulate multi-class classification accuracy. The accuracy is
   averaged over mini-batches. If the spatial dimension is not singleton, i.e.
   there are multiple labels for each data instance, then the accuracy is also
   averaged among the spatial dimension.

   .. attribute:: bottoms

      The blob names for prediction and labels (in that order).

   .. attribute:: dim

      Default ``-2`` (penultimate). Specify the dimension to operate on.
