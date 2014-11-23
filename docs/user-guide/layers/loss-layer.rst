Loss Layers
~~~~~~~~~~~

.. class:: MultinomialLogisticLossLayer

   The multinomial logistic loss is defined as :math:`\ell = -\log(x_g)`, where
   :math:`x_1,\ldots,x_C` are probabilities for each of the :math:`C` classes
   conditioned on the input data, and :math:`g` is the corresponding
   ground-truth category.

   The conditional probability blob should be of the shape :math:`(W,H,C,N)`,
   and the ground-truth blob should be of the shape :math:`(W,H,1,N)`. Typically
   there is only one label for each instance, so :math:`W=H=1`. The ground-truth
   should be a **zero-based** index in the range of :math:`0,\ldots,C-1`.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the conditional probability input blob, and the second one
      specifies the name for the ground-truth input blob.

.. class:: SoftmaxLossLayer

   This is essentially a combination of :class:`MultinomialLogisticLossLayer`
   and :class:`SoftmaxLayer`. The given predictions :math:`x_1,\ldots,x_C` for
   the :math:`C` classes are transformed with a softmax function

   .. math::

      \sigma(x_1,\ldots,x_C) = (\sigma_1,\ldots,\sigma_C) = \left(\frac{e^{x_1}}{\sum_j
      e^{x_j}},\ldots,\frac{e^{x_C}}{\sum_je^{x_j}}\right)

   which essentially turn the predictions into non-negative values with
   exponential function and then re-normalize to make them look like
   probabilties. Then the transformed values are used to compute the multinomial
   logsitic loss as

   .. math::

      \ell = -\log(\sigma_g)

   Here :math:`g` is the ground-truth label.

   The shapes of inputs is the same as :class:`MultinomialLogisticLossLayer`:
   the multi-class predictions are assumed to be along the channel dimension.

   The reason we provide a combined softmax loss layer instead using one softmax
   layer and one multinomial logistic layer is that the combined layer produces
   the back-propagation error in a more numerically robust way.

   .. math::

      \frac{\partial \ell}{\partial x_i} = \frac{e^{x_i}}{\sum_j e^{x_j}}
      - \delta_{ig} = \sigma_i - \delta_{ig}

   Here :math:`\delta_{ig}` is 1 if :math:`i=g`, and 0 otherwise.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the conditional probability input blob, and the second one
      specifies the name for the ground-truth input blob.


