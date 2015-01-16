Norm Constraints
================

Norm constraints is a more "direct" way of restricting the model complexity by
explicitly shrinking the parameters every *n* iterations if the norm of the
parameters exceeds a given threshold.

.. class:: NoCons

   No constraint is applied.

.. class:: L2Cons

   Constrain the Euclidean norm of parameters. Note that the threshold and shrinking
   are applied to *each parameter*. Specifically, for the filters parameter of
   a convolution layer, the threshold is applied to each filter. Similarly, for
   the weights parameter of an inner product layer, the threshold is applied to
   the weights corresponding to each single output dimension of the inner
   product layer. When the norm of the parameter exceed the threshold, it is
   scaled down to have exactly the norm specified in threshold.

   See the MNIST with dropout code in the ``examples`` directory for an example
   of how ``L2Cons`` is used.

   .. attribute:: threshold

      The norm threshold.

   .. attribute:: every_n_iter

      Defautl 1. Indicates the frequency of norm constraint application.

