Loss Layers
~~~~~~~~~~~

.. class:: HingeLossLayer

   Compute the hinge loss for binary classification problems:

   .. math::

      \frac{1}{N}\sum_{i=1}^N \max(1 - \mathbf{y}_i \cdot \hat{\mathbf{y}}_i, 0)

   Here :math:`N` is the batch-size, :math:`\mathbf{y}_i \in \{-1,1\}` is
   the ground-truth label of the :math:`i`-th sample, and
   :math:`\hat{\mathbf{y}}_i` is the corresponding prediction.

   .. attribute:: weight

      Default ``1.0``. Weight of this loss function. Could be useful when
      combining multiple loss functions in a network.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the prediction :math:`\hat{\mathbf{y}}`, and the second one
      specifies the name for the ground-truth :math:`\mathbf{y}`.

.. class:: MultinomialLogisticLossLayer

   The multinomial logistic loss is defined as :math:`\ell = -w_g\log(x_g)`, where
   :math:`x_1,\ldots,x_C` are probabilities for each of the :math:`C` classes
   conditioned on the input data, :math:`g` is the corresponding
   ground-truth category, and :math:`w_g` is the *weight* for the :math:`g`-th
   class (default 1, see bellow).

   If the conditional probability blob is of the shape ``(dim1, dim2, ...,
   dim_channel, ..., dimN)``, then the ground-truth blob should be of the shape
   ``(dim1, dim2, ..., 1, ..., dimN)``. Here ``dim_channel``, historically called
   the "channel" dimension, is the user specified tensor dimension to compute
   loss on. This general case allows to produce multiple labels for each
   sample. For the typical case where only one (multi-class) label is produced
   for one sample, the conditional probability blob is the shape ``(dim_channel,
   dim_num)`` and the ground-truth blob should be of the shape ``(1, dim_num)``.

   The ground-truth should be a **zero-based** index in the range of
   :math:`0,\ldots,C-1`.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the conditional probability input blob, and the second one
      specifies the name for the ground-truth input blob.

   .. attribute:: weight

      Default ``1.0``. Weight of this loss function. Could be useful when
      combining multiple loss functions in a network.

   .. attribute:: weights

      This can be used to specify weights for different classes. The following
      values are allowed

      * Empty array (default). This means each category should be equally
        weighted.
      * A 1D vector of length ``channels``. This defines weights for each
        category.
      * An (N-1)D tensor of the shape of a data point. In other words, the same
        shape as the prediction except that the last mini-batch dimension is
        removed. This is equivalent to the above case if the prediction is a 2D
        tensor of the shape ``channels``-by-``mini-batch``.
      * An ND tensor of the same shape as the prediction blob. This allows us to
        fully specify different weights for different data points in
        a mini-batch. See :class:`SoftlabelSoftmaxLossLayer`.

   .. attribute:: dim

      Default ``-2`` (penultimate). Specify the dimension to operate on.

   .. attribute:: normalize

      Indicating how weights should be normalized if given. The following values
      are allowed

      * ``:local`` (default): Normalize the weights locally at each location
        (w,h), across the channels.
      * ``:global``: Normalize the weights globally.
      * ``:no``: Do not normalize the weights.

      The weights normalization are done in a way that you get the same
      objective function when specifying *equal weights* for each class as when
      you do not specify any weights. In other words, the total sum of the
      weights are scaled to be equal to weights x height x channels. If you
      specify ``:no``, it is your responsibility to properly normalize the
      weights.

.. class:: SoftlabelSoftmaxLossLayer

   Like the :class:`SoftmaxLossLayer`, except that this deals with *soft
   labels*. For multiclass classification with :math:`K` categories, we call an integer
   value :math:`y\in\{0,\ldots,K-1\}` a *hard label*. In contrast, a soft label is
   a vector on the :math:`K`-dimensional simplex. In other words, a soft label
   specifies a probability distribution over all the :math:`K` categories, while
   a hard label is a special case where all the probability masses concentrates
   on one single category. In this case, this loss is basically computing the
   KL-divergence D(p||q), where p is the ground-truth softlabel, and q is the
   predicted distribution.

   .. attribute:: dim

      Default ``-2`` (penultimate). Specify the dimension to operate on.

   .. attribute:: weight

      Default ``1.0``. Weight of this loss function. Could be useful when
      combining multiple loss functions in a network.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the conditional probability input blob, and the second one
      specifies the name for the ground-truth (soft labels) input blob.

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

      \ell = -w_g \log(\sigma_g)

   Here :math:`g` is the ground-truth label, and :math:`w_g` is the weight for
   the :math:`g`-th category. See the document of :class:`MultinomialLogisticLossLayer` for more
   details on what the weights mean and how to specify them.

   The shapes of the inputs are the same as for the :class:`MultinomialLogisticLossLayer`:
   the multi-class predictions are assumed to be along the channel dimension.

   The reason we provide a combined softmax loss layer instead of using one softmax
   layer and one multinomial logistic layer is that the combined layer produces
   the back-propagation error in a more numerically robust way.

   .. math::

      \frac{\partial \ell}{\partial x_i} = w_g\left(\frac{e^{x_i}}{\sum_j e^{x_j}}
      - \delta_{ig}\right) = w_g\left(\sigma_i - \delta_{ig}\right)

   Here :math:`\delta_{ig}` is 1 if :math:`i=g`, and 0 otherwise.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the conditional probability input blob, and the second one
      specifies the name for the ground-truth input blob.

   .. attribute:: dim

      Default ``-2`` (penultimate). Specify the dimension to operate on. For
      a 4D vision tensor blob, the default value (penultimate) translates to the
      3rd tensor dimension, usually called the "channel" dimension.

   .. attribute:: weight

      Default ``1.0``. Weight of this loss function. Could be useful when
      combining multiple loss functions in a network.

   .. attribute::
      weights
      normalize

      Properties for the underlying :class:`MultinomialLogisticLossLayer`. See
      its documentation for details.

.. class:: SquareLossLayer

   Compute the square loss for real-valued regression problems:

   .. math::

      \frac{1}{2N}\sum_{i=1}^N \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|^2

   Here :math:`N` is the batch-size, :math:`\mathbf{y}_i` is the real-valued
   (vector or scalar) ground-truth label of the :math:`i`-th sample, and
   :math:`\hat{\mathbf{y}}_i` is the corresponding prediction.

   .. attribute:: weight

      Default ``1.0``. Weight of this loss function. Could be useful when
      combining multiple loss functions in a network.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the prediction :math:`\hat{\mathbf{y}}`, and the second one
      specifies the name for the ground-truth :math:`\mathbf{y}`.

.. class:: BinaryCrossEntropyLossLayer

   A simpler alternative to :class:`MultinomialLogisticLossLayer` for the
   special case of binary classification.

   .. math::

      -\frac{1}{N}\sum_{i=1}^N \log(p_i)y_i + \log(1-p_i)(1-y_i)

   Here :math:`N` is the batch-size, :math:`\mathbf{y}_i` is the ground-truth
   label of the :math:`i`-th sample, and :math:``p_i`` is the corresponding
   prediction.

   .. attribute:: weight

      Default ``1.0``. Weight of this loss function. Could be useful when
      combining multiple loss functions in a network.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the prediction :math:`\hat{\mathbf{y}}`, and the second one
      specifies the name for the binary ground-truth labels :math:`\mathbf{p}`.

.. class:: GaussianKLLossLayer

    Given two inputs *mu* and *sigma* of the same size representing the means
    and standard deviations of a diagonal multivariate Gaussian distribution, the
    loss is the Kullback-Leibler divergence from that to the standard Gaussian of
    the same dimension.

    Used in variational autoencoders, as in `Kingma & Welling 2013 <http://arxiv.org/abs/1312.6114>`_, as a form of regularization.

   .. math::
      D_{KL}(\mathcal{N}(\mathbf{\mu}, \mathrm{diag}(\mathbf{\sigma})) \Vert \mathcal{N}(\mathbf{0}, \mathbf{I}) )
      =  -\frac{1}{2}\left(\sum_{i=1}^N (\mu_i^2 + \sigma_i^2 - 2\log\sigma_i) - N\right)

   .. attribute:: weight

      Default ``1.0``. Weight of this loss function. Could be useful when
      combining multiple loss functions in a network.

   .. attribute:: bottoms

      Should be a vector containing two symbols. The first one specifies the
      name for the mean vector :math:`\mathbf{\mu}`, and the second one
      the vector of standard deviations :math:`\mathbf{\sigma}`.
