Alex’s CIFAR-10 tutorial in Mocha
=================================

This example is converted from `Caffe's CIFAR-10 tutorials
<http://caffe.berkeleyvision.org/gathered/examples/cifar10.html>`_, which was
originally built based on details from Alex Krizhevsky’s `cuda-convnet
<https://code.google.com/p/cuda-convnet2/>`_. In this example, we will
demonstrate how to translate a network definition in Caffe to Mocha, and train
the network to roughly reproduce the test error rate of 18% (without data
augmentation) as reported in `Alex Krizhevsky's website
<http://www.cs.toronto.edu/~kriz/cifar.html>`_.

The `CIFAR-10 dataset <http://www.cs.toronto.edu/~kriz/cifar.html>`_ is
a labeled subset of the `80 Million Tiny Images
<http://people.csail.mit.edu/torralba/tinyimages/>`_ dataset, containing 60,000
32x32 color images in 10 categories. They are split into 50,000 training images
and 10,000 test images. The number of samples are the same as in :doc:`the MNIST
example </tutorial/mnist>`. However, the images here are a bit larger and have
3 channels. As we will see soon, the network is also larger, with one extra
convolution and pooling and two local response normalization layers. It is
recommended to read :doc:`the MNIST tutorial </tutorial/mnist>` first, as we
will not repeat all details here.

Caffe's Tutorial and Code
-------------------------

Caffe's tutorial for CIFAR-10 can be found `on their website
<http://caffe.berkeleyvision.org/gathered/examples/cifar10.html>`_. The code
can be located in ``examples/cifar10`` under Caffe's source tree. The code
folder contains several different definitions of networks and solvers. The
filenames should be self-explanatory. The *quick* files corresponds to a smaller
network without local response normalization layers. These networks are documented in
Caffe's tutorial, according to which they obtain around 75% test accuracy.

We will be using the *full* models, which gives us around 81% test accuracy.
Caffe's definition of the full model can be found in the file
`cifar10_full_train_test.prototxt
<https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_train_test.prototxt>`_.
The training script is
`train_full.sh
<https://github.com/BVLC/caffe/blob/master/examples/cifar10/train_full.sh>`_,
which trains in 3 different stages with solvers defined in

#. `cifar10_full_solver.prototxt <https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_solver.prototxt>`_
#. `cifar10_full_solver_lr1.prototxt <https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_solver_lr1.prototxt>`_
#. `cifar10_full_solver_lr2.prototxt <https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_solver_lr2.prototxt>`_

respectively. This looks complicated. But if you compare the files, you will
find that the three stages are basically using the same solver configurations
except with a ten-fold learning rate decrease after each stage.

Preparing the Data
------------------

Looking at the data layer of Caffe's network definition, it uses a LevelDB
database as a data source. The LevelDB database is converted from the original
binary files downloaded from `the CIFAR-10 dataset's website
<http://www.cs.toronto.edu/~kriz/cifar.html>`_. Mocha does not support the LevelDB
database, so we will do the same thing: download the original binary files and
convert them into a Mocha-recognizable data format, in our case a HDF5 dataset. We have
provided a Julia script `convert.jl`_ [1]_. You can call `get-cifar10.sh
<https://github.com/pluskid/Mocha.jl/blob/master/examples/cifar10/get-cifar10.sh>`_
directly, which will automatically download the binary files, convert them to HDF5
and prepare text index files that point to the HDF5 datasets.

Notice in Caffe's data layer, a ``transform_param`` is specified with
a ``mean_file``. We could use Mocha's :doc:`data transformers
</user-guide/data-transformer>` to do the same thing. But since we need to
compute the data mean during data conversion, for simplicity, we also perform
mean subtraction when converting data to the HDF5 format. See `convert.jl`_ for
details. Please refer to the :doc:`user's guide </user-guide/layers/data-layer>`
for more details about the HDF5 data format that Mocha expects.

After converting the data, you should be ready to load the data in Mocha with
:class:`HDF5DataLayer`. We define two layers for training data and test data
separately, using the same batch size as in Caffe's model definition:

.. code-block:: julia

   data_tr_layer = HDF5DataLayer(name="data-train", source="data/train.txt", batch_size=100)
   data_tt_layer = HDF5DataLayer(name="data-test", source="data/test.txt", batch_size=100)

In order to share the definition of common computation layers, Caffe uses the
same file to define both the training and test networks, and uses *phases* to
include and exclude layers that are used only in the training or testing phase.
Mocha does not do this as the layers defined in Julia code are just Julia
objects. We will simply construct training and test nets with a different
subsets of those Julia layer objects.

.. _convert.jl: https://github.com/pluskid/Mocha.jl/blob/master/examples/cifar10/convert.jl

.. [1] All the CIFAR-10 example related code in Mocha can be found in the
   ``examples/cifar10`` directory under the source tree.

Computation and Loss Layers
---------------------------

Translating the computation layers should be straightforward. For example, the
``conv1`` layer is defined in Caffe as

.. code-block:: protobuf

   layers {
     name: "conv1"
     type: CONVOLUTION
     bottom: "data"
     top: "conv1"
     blobs_lr: 1
     blobs_lr: 2
     convolution_param {
       num_output: 32
       pad: 2
       kernel_size: 5
       stride: 1
       weight_filler {
         type: "gaussian"
         std: 0.0001
       }
       bias_filler {
         type: "constant"
       }
     }
   }

This translates to Mocha as:

.. code-block:: julia

   conv1_layer = ConvolutionLayer(name="conv1", n_filter=32, kernel=(5,5), pad=(2,2),
       stride=(1,1), filter_init=GaussianInitializer(std=0.0001),
       bottoms=[:data], tops=[:conv1])

.. Tip::

   * The ``pad``, ``kernel_size`` and ``stride`` parameters in Caffe means the same
     pad for both the *width* and *height* dimension unless specified explicitly.
     In Mocha, we always explicitly use a 2-tuple to specify the parameters for the
     two dimensions.
   * A *filler* in Caffe corresponds to an :doc:`initializer
     </user-guide/initializer>` in Mocha.
   * Mocha has a constant initializer (initialize to 0) for the bias by default, so
     we do not need to specify it explicitly.

The rest of the translated Mocha computation layers are listed here:

.. code-block:: julia

   pool1_layer = PoolingLayer(name="pool1", kernel=(3,3), stride=(2,2), neuron=Neurons.ReLU(),
       bottoms=[:conv1], tops=[:pool1])
   norm1_layer = LRNLayer(name="norm1", kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
       bottoms=[:pool1], tops=[:norm1])
   conv2_layer = ConvolutionLayer(name="conv2", n_filter=32, kernel=(5,5), pad=(2,2),
       stride=(1,1), filter_init=GaussianInitializer(std=0.01),
       bottoms=[:norm1], tops=[:conv2], neuron=Neurons.ReLU())
   pool2_layer = PoolingLayer(name="pool2", kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
       bottoms=[:conv2], tops=[:pool2])
   norm2_layer = LRNLayer(name="norm2", kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
       bottoms=[:pool2], tops=[:norm2])
   conv3_layer = ConvolutionLayer(name="conv3", n_filter=64, kernel=(5,5), pad=(2,2),
       stride=(1,1), filter_init=GaussianInitializer(std=0.01),
       bottoms=[:norm2], tops=[:conv3], neuron=Neurons.ReLU())
   pool3_layer = PoolingLayer(name="pool3", kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
       bottoms=[:conv3], tops=[:pool3])
   ip1_layer   = InnerProductLayer(name="ip1", output_dim=10, weight_init=GaussianInitializer(std=0.01),
       weight_regu=L2Regu(250), bottoms=[:pool3], tops=[:ip1])

You might have already noticed that Mocha does not have a ReLU layer.
Instead, ReLU, like Sigmoid, are treated as :doc:`neurons or activation
functions </user-guide/neuron>` attached to layers.

Constructing the Network
------------------------

In order to train the network, we need to define a loss layer. We also define an
accuracy layer to be used in the test network for us to see how our network
performs on the test dataset during training. Translating directly from Caffe's
definitions:

.. code-block:: julia

   loss_layer  = SoftmaxLossLayer(name="softmax", bottoms=[:ip1, :label])
   acc_layer   = AccuracyLayer(name="accuracy", bottoms=[:ip1, :label])

Next we collect the layers, and define a Mocha :class:`Net` on
a :class:`CuDNNBackend`. You can use :class:`CPUBackend` if no CUDA-compatible
GPU devices are available. But it will be much slower (see also
:doc:`/user-guide/backend`).

.. code-block:: julia

   common_layers = [conv1_layer, pool1_layer, norm1_layer, conv2_layer, pool2_layer, norm2_layer,
                    conv3_layer, pool3_layer, ip1_layer]

   backend = GPUBackend()
   init(backend)

   net = Net("CIFAR10-train", backend, [data_tr_layer, common_layers..., loss_layer])

Configuring the Solver
----------------------

The configuration for Caffe's solver looks like this

.. code-block:: protobuf

   # reduce learning rate after 120 epochs (60000 iters) by factor 0f 10
   # then another factor of 10 after 10 more epochs (5000 iters)

   # The train/test net protocol buffer definition
   net: "examples/cifar10/cifar10_full_train_test.prototxt"
   # test_iter specifies how many forward passes the test should carry out.
   # In the case of CIFAR10, we have test batch size 100 and 100 test iterations,
   # covering the full 10,000 testing images.
   test_iter: 100
   # Carry out testing every 1000 training iterations.
   test_interval: 1000
   # The base learning rate, momentum and the weight decay of the network.
   base_lr: 0.001
   momentum: 0.9
   weight_decay: 0.004
   # The learning rate policy
   lr_policy: "fixed"
   # Display every 200 iterations
   display: 200
   # The maximum number of iterations
   max_iter: 60000
   # snapshot intermediate results
   snapshot: 10000
   snapshot_prefix: "examples/cifar10/cifar10_full"
   # solver mode: CPU or GPU
   solver_mode: GPU

First of all, the learning rate is dropped by a factor of 10 twice [3]_. Caffe
implements this by having three solver configurations with different learning
rates for each stage. We could do the same thing for Mocha, but Mocha has
a staged learning policy that makes this easier:

.. code-block:: julia

   lr_policy = LRPolicy.Staged(
     (60000, LRPolicy.Fixed(0.001)),
     (5000, LRPolicy.Fixed(0.0001)),
     (5000, LRPolicy.Fixed(0.00001)),
   )
   method = SGD()
   solver_params = make_solver_parameters(method, max_iter=70000,
       regu_coef=0.004, momentum=0.9, lr_policy=lr_policy,
       load_from="snapshots")
   solver = Solver(method, solver_params)

The other parameters like regularization coefficient and momentum are directly
translated from Caffe's solver configuration. Progress reporting and automatic
snapshots can equivalently be done in Mocha as *coffee breaks* for the solver:

.. code-block:: julia

   # report training progress every 200 iterations
   add_coffee_break(solver, TrainingSummary(), every_n_iter=200)

   # save snapshots every 5000 iterations
   add_coffee_break(solver, Snapshot("snapshots"), every_n_iter=5000)

   # show performance on test data every 1000 iterations
   test_net = Net("CIFAR10-test", backend, [data_tt_layer, common_layers..., acc_layer])
   add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

.. [3] Looking at the Caffe solver configuration, I happily realized that I am
   not the only person in the world who sometimes mis-type o as 0. :P

Training
--------

Now we can start training by calling ``solve(solver, net)``. Depending on
different :doc:`backends </user-guide/backend>`, the training speed can vary.
Here are some sample training logs from my own test. Note this is **not**
a controlled comparison, just to get a rough feeling.

Pure Julia on CPU
~~~~~~~~~~~~~~~~~

The training is quite slow on a pure Julia backend. It takes about 15 minutes to
run every 200 iterations.

.. code-block:: text

   20-Nov 06:58:26:INFO:root:004600 :: TRAIN obj-val = 1.07695698
   20-Nov 07:13:25:INFO:root:004800 :: TRAIN obj-val = 1.06556938
   20-Nov 07:28:26:INFO:root:005000 :: TRAIN obj-val = 1.15177973
   20-Nov 07:30:35:INFO:root:
   20-Nov 07:30:35:INFO:root:## Performance on Validation Set
   20-Nov 07:30:35:INFO:root:---------------------------------------------------------
   20-Nov 07:30:35:INFO:root:  Accuracy (avg over 10000) = 62.8200%
   20-Nov 07:30:35:INFO:root:---------------------------------------------------------
   20-Nov 07:30:35:INFO:root:
   20-Nov 07:45:33:INFO:root:005200 :: TRAIN obj-val = 0.93760641
   20-Nov 08:00:30:INFO:root:005400 :: TRAIN obj-val = 0.95650533
   20-Nov 08:15:29:INFO:root:005600 :: TRAIN obj-val = 1.03291103
   20-Nov 08:30:21:INFO:root:005800 :: TRAIN obj-val = 1.01833960
   20-Nov 08:45:17:INFO:root:006000 :: TRAIN obj-val = 1.10167430
   20-Nov 08:47:27:INFO:root:
   20-Nov 08:47:27:INFO:root:## Performance on Validation Set
   20-Nov 08:47:27:INFO:root:---------------------------------------------------------
   20-Nov 08:47:27:INFO:root:  Accuracy (avg over 10000) = 64.7100%
   20-Nov 08:47:27:INFO:root:---------------------------------------------------------
   20-Nov 08:47:27:INFO:root:
   20-Nov 09:02:24:INFO:root:006200 :: TRAIN obj-val = 0.88323826


CPU with Native Extension
~~~~~~~~~~~~~~~~~~~~~~~~~

We enabled Mocha's native extension, but disabled OpenMP by setting the OMP
number of threads to 1:

.. code-block:: julia

   ENV["OMP_NUM_THREADS"] = 1
   blas_set_num_threads(1)

According to the log, it takes roughly 160 seconds to finish every 200
iterations.

.. code-block:: text

   20-Nov 09:29:10:INFO:root:000800 :: TRAIN obj-val = 1.46420457
   20-Nov 09:31:48:INFO:root:001000 :: TRAIN obj-val = 1.63248945
   20-Nov 09:32:22:INFO:root:
   20-Nov 09:32:22:INFO:root:## Performance on Validation Set
   20-Nov 09:32:22:INFO:root:---------------------------------------------------------
   20-Nov 09:32:22:INFO:root:  Accuracy (avg over 10000) = 44.4300%
   20-Nov 09:32:22:INFO:root:---------------------------------------------------------
   20-Nov 09:32:22:INFO:root:
   20-Nov 09:35:00:INFO:root:001200 :: TRAIN obj-val = 1.33312901
   20-Nov 09:37:38:INFO:root:001400 :: TRAIN obj-val = 1.40529397
   20-Nov 09:40:16:INFO:root:001600 :: TRAIN obj-val = 1.26366557
   20-Nov 09:42:54:INFO:root:001800 :: TRAIN obj-val = 1.29758151
   20-Nov 09:45:32:INFO:root:002000 :: TRAIN obj-val = 1.40923050
   20-Nov 09:46:06:INFO:root:
   20-Nov 09:46:06:INFO:root:## Performance on Validation Set
   20-Nov 09:46:06:INFO:root:---------------------------------------------------------
   20-Nov 09:46:06:INFO:root:  Accuracy (avg over 10000) = 51.0400%
   20-Nov 09:46:06:INFO:root:---------------------------------------------------------
   20-Nov 09:46:06:INFO:root:
   20-Nov 09:48:44:INFO:root:002200 :: TRAIN obj-val = 1.24579735
   20-Nov 09:51:22:INFO:root:002400 :: TRAIN obj-val = 1.22985339

We also tried to use multi-thread computing:

.. code-block:: julia

   ENV["OMP_NUM_THREADS"] = 16
   blas_set_num_threads(16)

By using 16 cores to compute, I got very slight improvement (which may well due
to external factors as I did not control the comparison environment at all),
with roughly 150 seconds every 200 iterations. I did not try multi-thread
computing with less or more threads.

.. code-block:: text

   20-Nov 10:29:34:INFO:root:002400 :: TRAIN obj-val = 1.25820349
   20-Nov 10:32:04:INFO:root:002600 :: TRAIN obj-val = 1.22480259
   20-Nov 10:34:32:INFO:root:002800 :: TRAIN obj-val = 1.25739809
   20-Nov 10:37:02:INFO:root:003000 :: TRAIN obj-val = 1.32196600
   20-Nov 10:37:36:INFO:root:
   20-Nov 10:37:36:INFO:root:## Performance on Validation Set
   20-Nov 10:37:36:INFO:root:---------------------------------------------------------
   20-Nov 10:37:36:INFO:root:  Accuracy (avg over 10000) = 56.4300%
   20-Nov 10:37:36:INFO:root:---------------------------------------------------------
   20-Nov 10:37:36:INFO:root:
   20-Nov 10:40:06:INFO:root:003200 :: TRAIN obj-val = 1.17503929
   20-Nov 10:42:40:INFO:root:003400 :: TRAIN obj-val = 1.13562913
   20-Nov 10:45:09:INFO:root:003600 :: TRAIN obj-val = 1.17141657
   20-Nov 10:47:40:INFO:root:003800 :: TRAIN obj-val = 1.20520208
   20-Nov 10:50:12:INFO:root:004000 :: TRAIN obj-val = 1.24686298
   20-Nov 10:50:47:INFO:root:
   20-Nov 10:50:47:INFO:root:## Performance on Validation Set
   20-Nov 10:50:47:INFO:root:---------------------------------------------------------
   20-Nov 10:50:47:INFO:root:  Accuracy (avg over 10000) = 59.4500%
   20-Nov 10:50:47:INFO:root:---------------------------------------------------------
   20-Nov 10:50:47:INFO:root:
   20-Nov 10:53:16:INFO:root:004200 :: TRAIN obj-val = 1.11022978
   20-Nov 10:55:49:INFO:root:004400 :: TRAIN obj-val = 1.04538457

CUDA with cuDNN
~~~~~~~~~~~~~~~

It takes only 5~6 seconds to finish every 200 iterations on the
``CuDNNBackend``.

.. code-block:: text

   22-Nov 15:04:47:INFO:root:048600 :: TRAIN obj-val = 0.53777266
   22-Nov 15:04:52:INFO:root:048800 :: TRAIN obj-val = 0.60837102
   22-Nov 15:04:58:INFO:root:049000 :: TRAIN obj-val = 0.79333639
   22-Nov 15:04:59:INFO:root:
   22-Nov 15:04:59:INFO:root:## Performance on Validation Set
   22-Nov 15:04:59:INFO:root:---------------------------------------------------------
   22-Nov 15:04:59:INFO:root:  Accuracy (avg over 10000) = 76.5900%
   22-Nov 15:04:59:INFO:root:---------------------------------------------------------
   22-Nov 15:04:59:INFO:root:
   22-Nov 15:05:04:INFO:root:049200 :: TRAIN obj-val = 0.62640750
   22-Nov 15:05:10:INFO:root:049400 :: TRAIN obj-val = 0.57287318
   22-Nov 15:05:15:INFO:root:049600 :: TRAIN obj-val = 0.53166425
   22-Nov 15:05:21:INFO:root:049800 :: TRAIN obj-val = 0.60679358
   22-Nov 15:05:26:INFO:root:050000 :: TRAIN obj-val = 0.79003465
   22-Nov 15:05:26:INFO:root:Saving snapshot to snapshot-050000.jld...
   22-Nov 15:05:26:DEBUG:root:Saving parameters for layer conv1
   22-Nov 15:05:26:DEBUG:root:Saving parameters for layer conv2
   22-Nov 15:05:26:DEBUG:root:Saving parameters for layer conv3
   22-Nov 15:05:26:DEBUG:root:Saving parameters for layer ip1
   22-Nov 15:05:27:INFO:root:
   22-Nov 15:05:27:INFO:root:## Performance on Validation Set
   22-Nov 15:05:27:INFO:root:---------------------------------------------------------
   22-Nov 15:05:27:INFO:root:  Accuracy (avg over 10000) = 76.5200%
   22-Nov 15:05:27:INFO:root:---------------------------------------------------------
   22-Nov 15:05:27:INFO:root:
   22-Nov 15:05:33:INFO:root:050200 :: TRAIN obj-val = 0.61519235
   22-Nov 15:05:38:INFO:root:050400 :: TRAIN obj-val = 0.57314044
