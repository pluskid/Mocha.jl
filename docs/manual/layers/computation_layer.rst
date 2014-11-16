Computation Layers
~~~~~~~~~~~~~~~~~~

Power Layer
-----------

Power layer performs element-wise operations as

.. math::

  O = (aI + b)^p

where :math:`a` is ``scale``, :math:`b` is ``shift``, and :math:`p` is
``power``. During back propagation, the following element-wise derivatives are
computed:

.. math::

  \frac{\partial O}{\partial I} = pa(aI + b)^{p-1}

Split Layer
-----------

Split layer produces identical *copies* [1]_ of the input. The number of copies
is determined by the length of the ``tops`` property. During back propagation,
derivatives from all the output copies are added together and propagated down.

This layer is typically used as a helper to implement some more complicated
layers.

.. [1] All the data is shared, so there is no actually data copying.
