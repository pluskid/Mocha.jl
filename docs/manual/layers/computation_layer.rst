Computation Layers
~~~~~~~~~~~~~~~~~~

.. class:: PoolingLayer

   2D pooling over the 2 image dimensions (width and height).

   .. attribute:: kernel

      Default (1,1), a 2-tuple of integers specifying pooling kernel width and
      height, respectively.

   .. attribute:: stride

      Default (1,1), a 2-tuple of integers specifying pooling stride in the
      width and height dimensions respectively.

   .. attribute:: pad

      Default (0,0), a 2-tuple of integers specifying the padding in the width and
      height dimensions respectively. Paddings are two-sided, so a pad of (1,0)
      will pad one pixel in both the left and the right boundary of an image.

   .. attribute:: pooling

      Default ``Pooling.Max()``. Specify the pooling operation to use.

   .. attribute::
      tops
      bottoms

      Blob names for output and input.

.. class:: ElementWiseLayer

   Element-wise layer implements basic element-wise operations on inputs.

   .. attribute:: operation

      Element-wise operation. Built-in operations are in module
      ``ElementWiseFunctors``, including ``Add``, ``Subtract``, ``Multiply`` and
      ``Divide``.

   .. attribute:: tops

      Output blob names, only one output blob is allowed.

   .. attribute:: bottoms

      Input blob names, count must match the number of inputs ``operation`` takes.

.. class:: PowerLayer

   Power layer performs element-wise operations as

   .. math::

     O = (aI + b)^p

   where :math:`a` is ``scale``, :math:`b` is ``shift``, and :math:`p` is
   ``power``. During back propagation, the following element-wise derivatives are
   computed:

   .. math::

     \frac{\partial O}{\partial I} = pa(aI + b)^{p-1}

   Power layer is implemented separately instead of as an Element-wise layer
   for better performance because there are some many special cases of Power layer that
   could be computed more efficiently.

   .. attribute:: power

      Default 1

   .. attribute:: scale

      Default 1

   .. attribute:: shift

      Default 0

   .. attribute::
      tops
      bottoms

      Blob names for output and input.

.. class:: SplitLayer

   Split layer produces identical *copies* [1]_ of the input. The number of copies
   is determined by the length of the ``tops`` property. During back propagation,
   derivatives from all the output copies are added together and propagated down.

   This layer is typically used as a helper to implement some more complicated
   layers.

   .. attribute:: bottoms

      Input blob names, only one input blob is allowed.

   .. attribute:: tops

      Output blob names, should be more than one output blobs.

   .. [1] All the data is shared, so there is no actually data copying.

.. class:: ChannelPoolingLayer

   1D pooling over the channel dimension.

   .. attribute:: kernel

      Default 1, pooling kernel size.

   .. attribute:: stride

      Default 1, stride for pooling.

   .. attribute:: pad

      Default (0,0), a 2-tuple specifying padding in the front and the end.

   .. attribute:: pooling

      Default ``Pooling.Max()``. Specify the pooling function to use.

   .. attribute::
      tops
      bottoms

      Blob names for output and input.

