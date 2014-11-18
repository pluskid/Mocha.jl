Regularizers
============

Regularizers add extra penalties or constraints for network parameters to
restrict the model complexity. The correspondences in Caffe are weight decays.
Regularizers and weight decays are equivalent in back-propagation. The
*conceptual* difference in the forward pass is that when treated as weight
decay, they are not considered as parts of the objective function. However, in
order to save computation, Mocha also omit forward computation for regularizers
by default. We choose to use the term regularization instead of weight decay
just because it is easier to understand when generalizing to sparse,
group-sparse or even more complicated structural regularizations.

All regularizers have the property ``coefficient``, corresponding to the
regularization coefficient. During training, a global regularization coefficient
can also be specified (see :doc:`user-guide/solver`), that globally scale all
local regularization coefficients.

.. class:: NoRegu

   Regularizer that impose no regularization.

.. class:: L2Regu

   L2 regularizer. The parameter blob :math:`W` is treated as a 1D vector.
   During the forward pass, the squared L2-norm :math:`\|W\|^2=\langle
   W,W\rangle` is computed, and :math:`\lambda \|W\|^2` is added to the
   objective function, where :math:`\lambda` is the regularization coefficient.
   During the backward pass, :math:`\lambda W` is added to the parameter
   gradient, enforcing a weight decay when the solver moves the parameters
   towards the negative gradient direction.
