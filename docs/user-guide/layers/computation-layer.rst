Computation Layers
~~~~~~~~~~~~~~~~~~

.. class:: ArgmaxLayer

   Compute the arg-max along the "channel" dimension. This layer is only used in
   the test network to produce predicted classes. It has no ability to do back
   propagation.

   .. attribute:: dim

      Default ``-2`` (penultimate). Specify which dimension to operate on.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same.

.. class:: ChannelPoolingLayer

   1D pooling over any specified dimension. This layer is called channel pooling layer
   because it was designed to pool over the pre-defined *channel* dimension back when Mocha could
   only handle 4D tensors. For the new, general ND-tensors the dimension to be pooled
   over can be freely specified by the user.

   .. attribute:: channel_dim

      Default ``-2`` (penultimate). Specifies which dimension to pool over.

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

      Blob names for output and input. This layer can take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same.

.. class:: ConvolutionLayer

   Convolution in the spatial dimensions. **For now** convolution layers
   require the input blobs to be 4D tensors. For a 4D input blob of the shape
   ``width``-by-``height``-by-``channels``-by-``num``, The output blob shape is decided by the
   ``kernel`` size (a.k.a. receptive field), the ``stride``, the ``pad`` and the
   ``n_filter``.

   The ``kernel`` size specifies the geometry of a filter, also called a kernel or a local
   receptive field. Note that implicitly, a filter also has a channel dimension that
   is the same size as the input image. As a filter moves across the image by
   the specified ``stride`` and optionally ``pad`` when on the boundary of the
   input image, it produce a real number by computing the inner-product between the
   filter weights and the local image patch at each spatial position. The
   formula for the spatial dimension of the output blob is

   .. code-block:: julia

      width_out  = div(width_in  + 2*pad[1]-kernel[1], stride[1]) + 1
      height_out = div(height_in + 2*pad[2]-kernel[2], stride[2]) + 1

   The ``n_filter`` parameter specifies the number of such filters. The final
   output blob will have the shape
   ``width_out``-by-``height_out``-by-``n_filter``-by-``num``. An illustration
   of typical convolution (and pooling) is shown below:

   .. image:: ../images/cnn-layer.*

   :sub:`Image credit:
   http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/`

   Here the *RF size* is *receptive field size*, and *maps* (identified by
   different colors) correspond to different filters.

   .. attribute:: param_key

      Default ``""``. The unique identifier for layers with shared parameters. When
      empty, the layer ``name`` is used as identifier instead.

   .. attribute:: kernel

      Default (1,1), a 2-tuple specifying the width and height of the
      convolution filters.

   .. attribute:: stride

      Default (1,1), a 2-tuple specifying the stride in the width and height
      dimensions, respectively.

   .. attribute:: pad

      Default (0,0), a 2-tuple specifying the two-sided padding in the width and
      height dimensions, respectively.

   .. attribute:: n_filter

      Default 1. Number of filters.

   .. attribute:: n_group

      Default 1. Number of groups. This number should divide both ``n_filter``
      and the number of channels in the input blob. This parameter will divide
      the input blob along the channel dimension into ``n_group`` groups. Each
      group will operate independently. Each group is assigned with ``n_filter``
      / ``n_group`` filters.

   .. attribute:: neuron

      Default ``Neurons.Identity()``, can be used to specify an activation
      function for the convolution outputs.

   .. attribute:: filter_init

      Default ``XavierInitializer()``. See :doc:`initializer
      </user-guide/initializer>` for the filters.

   .. attribute:: bias_init

      Default ``ConstantInitializer(0)``. See :doc:`initializer
      </user-guide/initializer>` for the bias.

   .. attribute:: filter_regu

      Default ``L2Regu(1)``, the regularizer for the filters.

   .. attribute:: bias_regu

      Default ``NoRegu()``, the regularizer for the bias.

   .. attribute:: filter_cons

      Default ``NoCons()``. :doc:`Norm constraint </user-guide/constraint>` for
      the filters.

   .. attribute:: bias_cons

      Default ``NoCons()``. Norm constraint for the bias. Typically no
      norm constraint should be applied to the bias.

   .. attribute:: filter_lr

      Default 1.0. The local learning rate for the filters.

   .. attribute:: bias_lr

      Default 2.0. The local learning rate for the bias.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input blobs
      and produce the corresponding number of output blobs. The shapes of the
      input blobs **must be the same**.

.. class:: CropLayer

   Do image cropping. This layer is primarily used only on top of data layers so
   backpropagation is currently not implemented. Crop layer requires the input
   blobs to be 4D tensors.

   .. attribute:: crop_size

      A (width, height) tuple of the size of the cropped image.

   .. attribute:: random_crop

      Default ``false``. When enabled, randomly place the cropping box instead
      of putting at the center. This is useful to produce random perturbations of
      the input images during training.

   .. attribute:: random_mirror

      Default ``false``. When enabled, randomly (with probability 0.5) mirror
      the input images (flip the width dimension).

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input blobs
      and produce the corresponding number of output blobs. The shapes of the
      input blobs do not need to be the same as long as they are valid (not
      smaller than the shape specified in ``crop_size``).

.. class:: DropoutLayer

   Dropout is typically used during training, and it has been demonstrated to be
   effective as a regularizer for large scale networks. Dropout operates by
   randomly "turning off" some responses. Specifically, the forward computation is

   .. math::

      y = \begin{cases}\frac{x}{1-p} & u > p \\ 0 & u <= p\end{cases}

   where :math:`u` is a random number uniformly distributed in [0,1], and
   :math:`p` is the ``ratio`` hyper-parameter. Note the output is scaled by
   :math:`1-p` such that :math:`\mathbb{E}[y] = x`.

   .. attribute:: ratio

      The probability :math:`p` of turning off a response. Can also be
      interpreted as the ratio of all the responses that are turned off.

   .. attribute:: auto_scale

      Default ``true``. When turned off, does not scale the result by
      :math:`1/(1-p)`. This option is used when building :class:`RandomMaskLayer`.

   .. attribute:: bottoms

      The names of the input blobs dropout operates on. Note this is a *in-place
      layer*, so

      1. there is no ``tops`` property. The output blobs will be the same as the
         input blobs.
      2. It takes **only one** input blob.

.. class:: ElementWiseLayer

   The Element-wise layer implements basic element-wise operations on inputs.

   .. attribute:: operation

      Element-wise operation. Built-in operations are defined in module
      ``ElementWiseFunctors``, including ``Add``, ``Subtract``, ``Multiply`` and
      ``Divide``.

   .. attribute:: tops

      Output blob names, only one output blob is allowed.

   .. attribute:: bottoms

      Input blob names, count must match the number of inputs ``operation`` takes.


.. class:: InnerProductLayer

   Densely connected linear layer. The output is computed as

   .. math::

      y_i = \sum_j w_{ij}x_j + b_i

   where :math:`w_{ij}` are the weights and :math:`b_i` are the biases.

   .. attribute:: param_key

      Default ``""``. The unique identifier for layers with shared parameters. When
      empty, the layer ``name`` is used as identifier instead.

   .. attribute:: output_dim

      Output dimension of the linear map. The input dimension is automatically
      decided via the inputs.

   .. attribute:: weight_init

      Default ``XavierInitializer()``. Specify how the weights :math:`w_{ij}` should
      be initialized.

   .. attribute:: bias_init

      Default ``ConstantInitializer(0)``, initializing the bias :math:`b_i`
      to 0.

   .. attribute:: weight_regu

      Default ``L2Regu(1)``. :doc:`Regularizer </user-guide/regularizer>` for the weights.

   .. attribute:: bias_regu

      Default ``NoRegu()``. Regularizer for the bias. Typically no
      regularization should be applied to the bias.

   .. attribute:: weight_cons

      Default ``NoCons()``. :doc:`Norm constraint </user-guide/constraint>` for the weights.

   .. attribute:: bias_cons

      Default ``NoCons()``. Norm constraint for the bias. Typically no
      norm constraint should be applied to the bias.

   .. attribute:: weight_lr

      Default 1.0. The local learning rate for the weights.

   .. attribute:: bias_lr

      Default 2.0. The local learning rate for the bias.

   .. attribute:: neuron

      Default ``Neurons.Identity()``, an optional :doc:`activation function
      </user-guide/neuron>` for the output of this layer.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input blobs
      and produce the corresponding number of output blobs. The feature
      dimensions (the product of the first N-1 dimensions) of all input blobs
      should be the same, but they could potentially have different batch sizes
      (the last dimension).

.. class:: LRNLayer

   Local Response Normalization Layer. It performs normalization over local
   input regions via the following mapping

   .. math::

      x \rightarrow y = \frac{x}{\left( \beta + (\alpha/n)\sum_{x_j\in N(x)}x_j^2
      \right)^p}

   Here :math:`\beta` is the shift, :math:`\alpha` is the scale, :math:`p` is
   the power, and :math:`n` is the size of the local neighborhood. :math:`N(x)`
   denotes the local neighborhood of :math:`x` of size :math:`n` (including
   :math:`x` itself). There are two types of local neighborhood:

   * ``LRNMode.AcrossChannel()``: The local neighborhood is a region of shape
     (1, 1, :math:`k`, 1) centered at :math:`x`. In other words, the region
     extends across nearby channels (with zero padding if needed), but has no
     spatial extent. Here :math:`k` is the kernel size, and :math:`n=k` in this
     case.
   * ``LRNMode.WithinChannel()``: The local neighborhood is a region of shape
     (:math:`k`, :math:`k`, 1, 1) centered at :math:`x`. In other words, the
     region extends spatially (in **both** the width and the channel dimension),
     again with zero padding when needed. But it does not extend across
     different channels. In this case :math:`n=k^2`.

     When this mode is used, the input blobs should be 4D tensors **for now**,
     due to the requirements from the underlying :class:`PoolingLayer`.

   .. attribute:: kernel

      Default 5, an integer indicating the kernel size. See :math:`k` in the
      descriptions above.

   .. attribute:: scale

      Default 1.

   .. attribute:: shift

      Default 1 (yes, 1, not 0).

   .. attribute:: power

      Default 0.75.

   .. attribute:: mode

      Default ``LRNMode.AcrossChannel()``.

   .. attribute::
      tops
      bottoms

      Names for output and input blobs. Only **one** input and **one** output blob are
      allowed.

.. class:: PoolingLayer

   2D pooling over the 2 image dimensions (width and height). **For now** the
   input blobs are required to be 4D tensors.

   .. attribute:: kernel

      Default (1,1), a 2-tuple of integers specifying pooling kernel width and
      height, respectively.

   .. attribute:: stride

      Default (1,1), a 2-tuple of integers specifying pooling stride in the
      width and height dimensions, respectively.

   .. attribute:: pad

      Default (0,0), a 2-tuple of integers specifying the padding in the width and
      height dimensions, respectively. Paddings are two-sided, so a pad of (1,0)
      will pad one pixel in both the left and the right boundary of an image.

   .. attribute:: pooling

      Default ``Pooling.Max()``. Specify the pooling operation to use.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same.

.. class:: PowerLayer

   Power layer performs element-wise operations as

   .. math::

     y = (ax + b)^p

   where :math:`a` is ``scale``, :math:`b` is ``shift``, and :math:`p` is
   ``power``. During back propagation, the following element-wise derivatives are
   computed:

   .. math::

     \frac{\partial y}{\partial x} = pa(ax + b)^{p-1}

   Power layer is implemented separately instead of as an Element-wise layer
   for better performance because there are some special cases of the Power layer that
   can be computed more efficiently.

   .. attribute:: power

      Default 1

   .. attribute:: scale

      Default 1

   .. attribute:: shift

      Default 0

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same.

.. class:: RandomMaskLayer

   Randomly mask subsets of input as zero. This is a wrapper over
   :class:`DropoutLayer`, but

   * This layer does not rescale the un-masked part to make the expectation the
     same as the expectation of the original input.
   * This layer can handle multiple input blobs while :class:`DropoutLayer`
     accept only one input blob.

   .. note::

      * This layer is a in-place layer. For example, if you want to use this to
        construct a denoising auto-encoder, you should use a :class:`SplitLayer`
        to make two copies of the input data: one is randomly masked (in-place)
        as the input of the auto-encoder, and the other is directed to
        a :class:`SquareLoss` layer that measure the reconstruction error.
      * Although typically not used, this layer is capable of doing
        back-propagation, powered by the underlying :class:`DropoutLayer`.

.. class:: SoftmaxLayer

   Compute softmax over the "channel" dimension. The inputs :math:`x_1,\ldots,x_C`
   are mapped as

   .. math::

      \sigma(x_1,\ldots,x_C) = (\sigma_1,\ldots,\sigma_C) = \left(\frac{e^{x_1}}{\sum_j
      e^{x_j}},\ldots,\frac{e^{x_C}}{\sum_je^{x_j}}\right)

   To train a multi-class classification network with softmax probability output
   and multiclass logistic loss, use the bundled :class:`SoftmaxLossLayer`
   instead.

   .. attribute:: dim

      Default ``-2`` (penultimate). Specify the "channel" dim to operate along.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input
      blobs and produce the corresponding number of output blobs. The shapes of
      the input blobs do not need to be the same.

.. class:: TiedInnerProductLayer

   Similar to :class:`InnerProductLayer` but with *weights tied* to an existing
   :class:`InnerProductLayer`. Used in auto-encoders. During training, an
   auto-encoder defines the following mapping

   .. math::

      \mathbf{x} \longrightarrow \mathbf{h} = \mathbf{W}_1^T\mathbf{x}
      + \mathbf{b}_1 \longrightarrow \tilde{\mathbf{x}} = \mathbf{W}_2^T\mathbf{h}
        + \mathbf{b}_2

   Here :math:`\mathbf{x}` is input, :math:`\mathbf{h}` is the latent encoding, and
   :math:`\tilde{\mathbf{x}}` is the decoded reconstruction of the input. Sometimes
   it is desired to have *tied weights* for the encoder and decoder:
   :math:`\mathbf{W}_1 = \mathbf{W}^T`. In this case, the encoder will be an
   :class:`InnerProductLayer`, and the decoder a :class:`TiedInnerProductLayer`
   with tied weights to the encoder layer.

   Note the tied decoder layer does *not* perform learning for the weights.
   However, even a tied layer has independent bias parameters that are learned
   independently.

   .. attribute:: tied_param_key

      The ``param_key`` of the encoder layer that this layer wants to share tied
      weights with.

   .. attribute:: param_key

      Default ``""``. The unique identifier for layers with shared parameters.
      If empty, the layer ``name`` is used as identifier instead.

      .. tip::

         * ``param_key`` is used for :class:`TiedInnerProductLayer` to share
           parameters. For example, the same layer in a training net and in
           a validation / testing net use this mechanism to share parameters.
         * ``tied_param_key`` is used to find the :class:`InnerProductLayer` to
           enable *tied weights*. This should be equal to the ``param_key``
           property of the inner product layer you want to have tied weights
           with.

   .. attribute:: bias_init

      Default ``ConstantInitializer(0)``. The :doc:`initializer
      </user-guide/initializer>` for the bias.

   .. attribute:: bias_regu

      Default ``NoRegu()``, the regularizer for the bias.

   .. attribute:: bias_cons

      Default ``NoCons()``. Norm constraint for the bias. Typically no
      norm constraint should be applied to the bias.

   .. attribute:: bias_lr

      Default 2.0. The local learning rate for the bias.

   .. attribute:: neuron

      Default ``Neurons.Identity()``, an optional :doc:`activation function
      </user-guide/neuron>` for the output of this layer.

   .. attribute::
      tops
      bottoms

      Blob names for output and input. This layer can take multiple input blobs
      and produce the corresponding number of output blobs. The feature
      dimensions (the product of the first N-1 dimensions) of all input blobs
      should be the same, but they can potentially have different batch sizes
      (the last dimension).


.. class:: RandomNormalLayer

    This is a source layer which outputs standard Gaussian random noise.

   .. attribute:: tops

      List of symbols, specifying the names of the noise blobs to produce.

   .. attribute:: output_dims

      List of integers giving the dimensions of the output noise blobs.

   .. attribute:: batch_sizes

      List of integers the same length as ``tops``, giving the number of vectors
      to output in each batch.

   .. attribute:: eltype

      Default ``Float32``.
