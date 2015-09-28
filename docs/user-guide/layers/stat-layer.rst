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

      Default ``-2`` (penultimate). Specifies the dimension to operate on.

.. class:: BinaryAccuracyLayer

   Compute and accumulate binary classification accuracy. The accuracy is
   averaged over mini-batches. Labels can be either {0, 1} labels or
   {-1, +1} labels

   .. attribute:: bottoms

      The blob names for prediction and labels (in that order).

   .. attribute: threshold

      Thresholds the predictions and labels. Use ``0.5`` if labels are from
      the set {0, 1}, or ``0`` if labels are from the set {-1, +1}.
