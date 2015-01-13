Regularizers
============

Regularizers add extra penalties or constraints for network parameters to
restrict the model complexity. The corresponding term used in Caffe is *weight decay*.
Regularization and weight decay are equivalent in back-propagation. The
*conceptual* difference in the forward pass is that when treated as weight
decay, they are not considered being part of the objective function. However, in
order to reduce the number of computations, Mocha also omits the forward computation for regularizers
by default. We choose to use the term regularization instead of weight decay
just because it is easier to understand when generalizing to sparse,
group-sparse or even more complicated structural regularizations.

All regularizers have the property ``coefficient``, corresponding to the
regularization coefficient. During training, a global regularization coefficient
can also be specified (see :doc:`user-guide/solver`), which globally scales all
local regularization coefficients.

.. class:: NoRegu

   Regularizer that imposes no regularization.

.. class:: L2Regu

   L2 regularizer. The parameter blob :math:`W` is treated as a 1D vector.
   During the forward pass, the squared L2-norm :math:`\|W\|^2=\langle
   W,W\rangle` is computed, and :math:`\lambda \|W\|^2` is added to the
   objective function, where :math:`\lambda` is the regularization coefficient.
   During the backward pass, :math:`2\lambda W` is added to the parameter
   gradient, enforcing a weight decay when the solver moves the parameters
   towards the negative gradient direction.

   .. note::

      In Caffe, only :math:`\lambda W` is added as a weight decay in back propagation,
      which is equivalent to having a L2 regularizer with coefficient
      :math:`0.5\lambda`.

.. class:: L1Regu

   L1 regularizer. The parameter blob :math:`W` is treated as a 1D vector.
   During the forward pass, the L1-norm

   .. math::

      \|W\|_1 = \sum_i |W_i|

   is computed, and :math:`\lambda \|W\|_1` is added to the objective function.
   During the backward pass, :math:`\lambda\text{sign}(W)` is added to the
   parameter gradient. The L1 regularizer has the property of encouraging
   sparsity in the parameters.

