Pre-training with Stacked De-noising Auto-encoders
==================================================

In this tutorial, we show how to use Mocha's primitives to build stacked
auto-encoders to do pre-training for a deep neural network. We will work with
the MNIST dataset. Please see :doc:`the LeNet tutorial on
MNIST </tutorial/mnist>` on how to prepare the HDF5 dataset.

Unsupervised pre-training is a way to initialize the weights when training
a deep neural networks. Initialization with pre-training could have better
convergence property than simple random training, especially when the number of
(labeled) training points is not very large.

In the following two figures, we show the results generated from this tutorial.
Specifically, the first figure shows the softmax loss on the training set at
different training iterations with and without pre-training initialization.

.. image:: images/mnist-sDA/obj-val.*

The second plot is similar, except that it shows the prediction accuracy of the
trained model on the test set.

.. image:: images/mnist-sDA/test-accuracy-accuracy.*

As we can see, faster convergence could be observed when initialize with
pre-training.

(stacked) Denoising Auto-encoders
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

Note the bias :math:`\mathbf{b}` and :math:`\tilde{\mathbf{b}}` are still
different even when the weights are *tied*. An auto-encoder is trained by
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
of stacked (denoising) auto-encoders could be used to obtain initializations for
all the hidden layers.

Layer-wise pre-training of stacked auto-encoders consists of the following
procedures:

1. Train the bottom most auto-encoder.
2. After training, remove the decoder layer, construct a new auto-encoder by
   taking the *latent representation* of existing auto-encoder as input.
3. Train the new auto-encoder. Note the weights and bias of the encoder from the
   previously trained auto-encoders are **fixed** when training the newly
   constructed auto-encoder.
4. Repeat step 2 and 3 until enough layers pre-trained.

Next we will show how to train denoising auto-encoders in Mocha and use them to
initialize DNNs.

Experiment Configuration
------------------------

We will train a DNN with 3 hidden layers using sigmoid nonlinearity. All the
parameters are listed below:

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-config--
   :end-before: --end-config--

As we can see, we will do 15 epochs when pre-training for each layer, and do
1000 epochs of fine-tuning.

In Mocha, parameters (weights and bias) could be shared among different layers
by specifying the ``param_key`` parameter when constructing layers. The
``param_keys`` variable defined above are unique identifiers for each of
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

.. literalinclude:: ../../examples/unsupervised-pretrain/denoising-autoencoder/denoising-autoencoder.jl
   :start-after: --start-pre-train--
   :end-before: --end-pre-train--
