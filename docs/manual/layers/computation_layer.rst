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
