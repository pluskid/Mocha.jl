Pre-training with Stacked De-noising Auto-encoders
==================================================

In this tutorial, we show how to use Mocha's primitives to build stacked
auto-encoders to do pre-training for a deep neural network. We will work with
the MNIST dataset. Please see :doc:`the LeNet tutorial on
MNIST </tutorial/mnist>` on how to prepare the HDF5 dataset.

Unsupervised pre-training is a way to initialize the weights when training
deep neural networks. Initialization with pre-training can have better
convergence properties than simple random training, especially when the number of
(labeled) training points is not very large.

In the following two figures, we show the results generated from this tutorial.
Specifically, the first figure shows the softmax loss on the training set at
different training iterations with and without pre-training initialization.

.. image:: images/mnist-sDA/obj-val.*

The second plot is similar, except that it shows the prediction accuracy of the
trained model on the test set.

.. image:: images/mnist-sDA/test-accuracy-accuracy.*

As we can see, faster convergence can be observed when we initialize with
pre-training.

(Stacked) Denoising Auto-encoders
---------------------------------

We provide a brief introduction to (stacked) denoising auto-encoders in this
section. See also the `deep learning tutorial on Denoising Auto-encoders
<http://deeplearning.net/tutorial/dA.html>`_.

An **auto-encoder** takes an input :math:`\mathbf{x}\in \mathbb{R}^p`, maps it to
a latent representation (encoding) :math:`\mathbf{y}\in\mathbb{R}^q`, and then
maps back to the original space :math:`\mathbf{z}\in\mathbb{R}^p` (decoding
/ reconstruction). The mappings are typically linear maps (optionally) followed
by a element-wise nonlinearity:

.. math::

   \begin{aligned}
   \mathbf{y} &= s\left(\mathbf{W}\mathbf{x} + \mathbf{b}\right) \\
   \mathbf{z} &= s\left(\tilde{\mathbf{W}}\mathbf{y} + \tilde{\mathbf{b}}\right)
   \end{aligned}

Typically, we constrain the weights in the decoder to be the transpose of the
weights in the encoder. This is referred to as *tied weights*:

.. math::

   \tilde{\mathbf{W}} = \mathbf{W}^T

Note that the biases :math:`\mathbf{b}` and :math:`\tilde{\mathbf{b}}` are still
different even when the weights are tied. An auto-encoder is trained by
minimizing the reconstruction error, typically with the square loss
:math:`\ell(\mathbf{x},\mathbf{z})=\|\mathbf{x}-\mathbf{z}\|^2`.

A **denoising auto-encoder** is an auto-encoder with noise corruptions. More
specifically, the encoder takes a corrupted version :math:`\tilde{\mathbf{x}}`
of the original input. A typical way of corruption is randomly masking elements of
:math:`\mathbf{x}` as zeros. Note the reconstruction error is still measured
against the original uncorrupted input :math:`\mathbf{x}`.

After training, we can take the weights and bias of the encoder layer in
a (denoising) auto-encoder as an initialization of an hidden (inner-product)
layer of a DNN. When there are multiple hidden layers, layer-wise pre-training
of stacked (denoising) auto-encoders can be used to obtain initializations for
all the hidden layers.

Layer-wise pre-training of stacked auto-encoders consists of the following
steps:

1. Train the bottommost auto-encoder.
2. After training, remove the decoder layer, construct a new auto-encoder by
   taking the *latent representation* of the previous auto-encoder as input.
3. Train the new auto-encoder. Note the weights and bias of the encoder from the
   previously trained auto-encoders are **fixed** when training the newly
   constructed auto-encoder.
4. Repeat step 2 and 3 until enough layers are pre-trained.

Next we will show how to train denoising auto-encoders in Mocha and use them to
initialize DNNs.

Experiment Configuration
------------------------

We will train a DNN with 3 hidden layers using sigmoid nonlinearities. All the
parameters are listed below:

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-config--
   :end-before: --end-config--

As we can see, we will do 15 epochs when pre-training for each layer, and do
1000 epochs of fine-tuning.

In Mocha, parameters (weights and bias) can be shared among different layers
by specifying the ``param_key`` parameter when constructing layers. The
``param_keys`` variables defined above are unique identifiers for each of
the hidden layers. We will use those identifiers to indicate that the encoders
in pre-training share parameters with the hidden layers in DNN fine-tuning.

Here we define several basic layers that will be used in both pre-training and
fine-tuning.

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-basic-layers--
   :end-before: --end-basic-layers--

Note the ``rename_layer`` is defined to rename the ``:data`` blob to ``:ip0``
blob. This makes it easier to define the hidden layers in a unified manner.

Pre-training
------------

We construct stacked denoising auto-encoders to perform pre-training for the weights and
biases of the hidden layers we just defined. We do layer-wise pre-training in
a ``for`` loop. Several Mocha primitives are useful for building auto-encoders:

* :class:`RandomMaskLayer`: given a corruption ratio, this layer can randomly
  mask parts of the input blobs as zero. We use this to create corruptions in
  denoising auto-encoders.

  Note this is a *in-place* layer. In other words, it modifies the input
  directly. Recall that the reconstruction error is computed against the
  *uncorruppted* input. So we need to use the following layer to create a copy
  of the input before applying corruption.
* :class:`SplitLayer`: split a blob into multiple copies.
* :class:`InnerProductLayer`: the encoder layer is just an ordinary
  inner-product layer in DNNs.
* :class:`TiedInnerProductLayer`: if we do not want *tied weights*, we could use
  another inner-product layer as the decoder. Here we use a special layer to
  construct decoders with *tied weights*. The ``tied_param_key`` attribute is
  used to identify the corresponding encoder layer we want to tie weights with.
* :class:`SquareLossLayer`: used to compute reconstruction error.

We list the code for the layer definitions of the auto-encoders again:

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-sda-layers--
   :end-before: --end-sda-layers--

Note how the i-th auto-encoder is built on top of the output of the (i-1)-th hidden
layer (blob name ``symbol("ip$(i-1)")``). We split the blob into ``:orig_data``
and ``:corrupt_data``, and add corruption to the ``:corrupt_data`` blob.

The encoder layer is basically the same as the i-th hidden layer. But it should
take the corrupted blob as input, so use the ``copy`` function to make a new
layer based on the i-th hidden layer but change the ``bottoms`` property. The
decoder layer has *tied weights* with the encoder layer, and the square-loss
layer compute the reconstruction error.

Recall that in layer-wise pre-training, we fix the parameters of the encoder
layers that we already trained, and only train the top-most encoder-decoder
pair. In Mocha, we can *freeze* layers in a net to prevent their parameters
being modified during training. In this case, we freeze all layers except the
encoder and the decoder layers:

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-freeze--
   :end-before: --end-freeze--

Now we are ready to do the pre-training. In this example, we do not use
regularization or momentum:

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-pre-train--
   :end-before: --end-pre-train--

Fine Tuning
-----------

After pre-training, we are now ready to do supervised fine tuning. This part is
almost identical to the original :doc:`MNIST tutorial </tutorial/mnist>`.

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-finetune--
   :end-before: --end-finetune--

Note that the key to allow the ``MNIST-finetune`` net to use the pre-trained weights
as initialization of the hidden layers is that we specify the same ``param_key``
property for the hidden layers and the encoder layers. Those parameters are
stored in the registry of the ``backend``. When a net is constructed, if
a layer finds existing parameters with its ``param_key``, it will use the
existing parameters, and ignore the :doc:`parameter initializers
</user-guide/initializer>` specified by the user. Debug information will be
printed to the console:

.. code-block:: text

   31-Dec 02:37:46:DEBUG:root:InnerProductLayer(ip-1): sharing weights and bias
   31-Dec 02:37:46:DEBUG:root:InnerProductLayer(ip-2): sharing weights and bias
   31-Dec 02:37:46:DEBUG:root:InnerProductLayer(ip-3): sharing weights and bias


Comparison with Random Initialization
-------------------------------------

In order to see whether pre-training is helpful, we train the same DNN but with
random initialization. The same layer definitions are re-used. But note the
highlighted line below: we reset the registry in the backend to clear the
pre-trained parameters before constructing the net:

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-randinit--
   :end-before: --end-randinit--
   :emphasize-lines: 1

We can check from the log that randomly initialized parameters are used in this
case:

.. code-block:: text

   31-Dec 01:55:06:DEBUG:root:Init network MNIST-rnd
   31-Dec 01:55:06:DEBUG:root:Init parameter weight for layer ip-1
   31-Dec 01:55:06:DEBUG:root:Init parameter bias for layer ip-1
   31-Dec 01:55:06:DEBUG:root:Init parameter weight for layer ip-2
   31-Dec 01:55:06:DEBUG:root:Init parameter bias for layer ip-2
   31-Dec 01:55:06:DEBUG:root:Init parameter weight for layer ip-3
   31-Dec 01:55:06:DEBUG:root:Init parameter bias for layer ip-3
   31-Dec 01:55:06:DEBUG:root:Init parameter weight for layer pred
   31-Dec 01:55:06:DEBUG:root:Init parameter bias for layer pred

The plots shown at the beginning of this tutorial are generated from the saved
statistics from the *coffee lounges*. If you are interested in how those plots
are generated, please refer to the ``plot-all.jl`` script in the code directory of this
tutorial.
