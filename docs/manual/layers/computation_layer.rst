Computation Layers
~~~~~~~~~~~~~~~~~~

Element-wise Layer
------------------

Element-wise layer implements basic element-wise operations on inputs.

Type name
  ``ElementWiseLayer``
Properties
  * ``operation``: Element-wise operation. Built-in operations are in module
    ``ElementWiseFunctors``, including ``Add``, ``Subtract``, ``Multiply`` and
    ``Divide``.
  * ``tops``: Output blob names, only one output blob is allowed.
  * ``bottoms``:: Input blob names, count must match the number of inputs
    ``operation`` takes.

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

Power layer is implemented separately instead of as an Element-wise layer
for better performance because there are some many special cases of Power layer that
could be computed more efficiently.

Type name
  ``PowerLayer``
Properties
  * ``power``: Default 1.
  * ``scale``: Default 1.
  * ``shift``: Default 0.
  * ``tops`` and ``bottoms``: Blob names for output and input.

Split Layer
-----------

Split layer produces identical *copies* [1]_ of the input. The number of copies
is determined by the length of the ``tops`` property. During back propagation,
derivatives from all the output copies are added together and propagated down.

This layer is typically used as a helper to implement some more complicated
layers.

Type name
  ``SplitLayer``
Properties
  * ``bottoms``: Input blob names, only one input blob is allowed.
  * ``tops``: Output blob names, should be more than one output blobs.

.. [1] All the data is shared, so there is no actually data copying.
