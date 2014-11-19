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
and 10,000 test images. The number of samples are the same to :doc:`the MNIST
example </tutorial/mnist>`. However, the images here are a bit larger and have
3 channels. As we will see soon, the network is also larger, with one extra
convolution and pooling and two local response normalization layers. It is
recommended to read :doc:`the MNIST tutorial </tutorial/mnist>` first, as we
will not repeat many details here.

Caffe's Tutorial and Code
-------------------------

Caffe's tutorial for CIFAR-10 can be found `on their website
<http://caffe.berkeleyvision.org/gathered/examples/cifar10.html>`_. The code
could be located in ``examples/cifar10`` under Caffe's source tree. The code
folder contains several different definition of networks and solvers. The
filenames should be self-explanatory. The *quick* files corresponds to a smaller
network without local response normalization layers. And this is documented in
Caffe's tutorial, according to which, produces around 75% test accuracy.

We will be using the *full* models, which gives us around 81% test accuracy.
Caffe's definition of the full model could be found in the file
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

Prepare the Data
----------------

Looking at the data layer of Caffe's network definition, it uses a LevelDB
database as a data source. The LevelDB database is converted from the original
binary files downloaded from `the CIFAR-10 dataset's website
<http://www.cs.toronto.edu/~kriz/cifar.html>`_. Mocha does not support LevelDB
database, so we will do the same thing: download the original binary files and
convert into a Mocha-recognizable data format, HDF5 dataset here. We have
provided a Julia script `convert.jl`_ [1]_. You can call `get-cifar10.sh
<https://github.com/pluskid/Mocha.jl/blob/master/examples/cifar10/get-cifar10.sh>`_
directly, which will automatically download the binary files, convert it to HDF5
and prepare text index files that points to the HDF5 datasets.

Notice in Caffe's data layer, a ``transform_param`` is specified with
a ``mean_file``. Mocha's data layers does not support data transformations now.
The philosophy is that unless you are doing massive data augmentation [2]_,
it is better to do those data preprocessing off-line. Since pre-processing only
needs to run once, being able to perform arbitrarily flexible manipulations is
more important than being super efficient.

In this case, we perform mean subtraction when we convert to the HDF5 dataset.
See `convert.jl`_ for details. Please refer to the :doc:`user's guide
</user-guide/layers/data-layer>` for more details about HDF5 data format that
Mocha reads.

After converting the data, you should be ready to load the data in Mocha with
:class:`HDF5DataLayer`. We define two layers for training data and test data
separately, using the same batch size as in Caffe's model definition:

.. code-block:: julia

   data_tr_layer = HDF5DataLayer(source="data/train.txt", batch_size=100)
   data_tt_layer = HDF5DataLayer(source="data/test.txt", batch_size=100)

In order to share the definition of common computation layers, Caffe use the
same file to define both the training and test networks, and use *phase* to
include and exclude layers that are used only in training or testing phases.
Mocha does not need this as the layers defined in Julia code are just Julia
objects. We will simply construct training and test nets with a different
subsets of those Julia layer objects.

.. _convert.jl: https://github.com/pluskid/Mocha.jl/blob/master/examples/cifar10/convert.jl

.. [1] All the CIFAR-10 example related code in Mocha could be found in the
   ``examples/cifar10`` directory under the source tree.
.. [2] And you are lack of disk spaces. Or you are generating a huge amount of
   extra transformed data with very cheap operations, such that loading
   pre-generated data is much slower than generating them on the fly.

Define Computation and Loss Layers
----------------------------------

Translating the computation layers should be straightforward. For example, the
``conv1`` layer is defined in Caffe as

.. code-block:: text

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

Several remarks:

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

   pool1_layer = PoolingLayer(kernel=(3,3), stride=(2,2), neuron=Neurons.ReLU(),
       bottoms=[:conv1], tops=[:pool1])
   norm1_layer = LRNLayer(kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
       bottoms=[:pool1], tops=[:norm1])
   conv2_layer = ConvolutionLayer(name="conv2", n_filter=32, kernel=(5,5), pad=(2,2),
       stride=(1,1), filter_init=GaussianInitializer(std=0.01),
       bottoms=[:norm1], tops=[:conv2], neuron=Neurons.ReLU())
   pool2_layer = PoolingLayer(kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
       bottoms=[:conv2], tops=[:pool2])
   norm2_layer = LRNLayer(kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
       bottoms=[:pool2], tops=[:norm2])
   conv3_layer = ConvolutionLayer(name="conv3", n_filter=64, kernel=(5,5), pad=(2,2),
       stride=(1,1), filter_init=GaussianInitializer(std=0.01),
       bottoms=[:norm2], tops=[:conv3], neuron=Neurons.ReLU())
   pool3_layer = PoolingLayer(kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
       bottoms=[:conv3], tops=[:pool3])
   ip1_layer   = InnerProductLayer(output_dim=10, weight_init=GaussianInitializer(std=0.01),
       weight_regu=L2Regu(250), bottoms=[:pool3], tops=[:ip1])

Another remark that you might have already noticed is that Mocha does not have
a ReLU layer. Instead, ReLU, like Sigmoid, are treated as :doc:`neurons or
activation functions </user-guide/neuron>` attached to layers.

Constructing the Network
------------------------

In order to train the network, we need to define a loss layer. We also define an
accuracy layer to be used in the test network for us to see how our network
performs on the test dataset during training. Translating directly from Caffe's
definitions:

.. code-block:: julia

   loss_layer  = SoftmaxLossLayer(bottoms=[:ip1, :label])
   acc_layer   = AccuracyLayer(bottoms=[:ip1, :label])

Next we collect the layers, and define a Mocha :class:`Net` on
a :class:`CuDNNBackend`. You could use :class:`CPUBackend` if no CUDA-compatible
GPU devices are available. But it will be much slower (see also
:doc:`/user-guide/backend`).

Configuring the Solver
----------------------



.. code-block:: julia

   common_layers = [conv1_layer, pool1_layer, norm1_layer, conv2_layer, pool2_layer, norm2_layer,
                    conv3_layer, pool3_layer, ip1_layer]

   sys = System(CuDNNBackend())
   #sys = System(CPUBackend())
   init(sys)

   net = Net(sys, [data_tr_layer, common_layers..., loss_layer])

.. code-block:: julia

   ENV["OMP_NUM_THREADS"] = 1
   blas_set_num_threads(1)

.. code-block:: text

   17-Nov 23:16:48:INFO:root:002800 :: TRAIN obj-val = 0.81475013
   17-Nov 23:18:53:INFO:root:003000 :: TRAIN obj-val = 0.96854031
   17-Nov 23:19:21:INFO:root:
   17-Nov 23:19:21:INFO:root:## Performance on Validation Set
   17-Nov 23:19:21:INFO:root:---------------------------------------------------------
   17-Nov 23:19:21:INFO:root:  Accuracy (avg over 10000) = 67.3000%
   17-Nov 23:19:21:INFO:root:---------------------------------------------------------
   17-Nov 23:19:21:INFO:root:
   17-Nov 23:21:27:INFO:root:003200 :: TRAIN obj-val = 1.09695852
   17-Nov 23:23:36:INFO:root:003400 :: TRAIN obj-val = 0.98007375
   17-Nov 23:25:49:INFO:root:003600 :: TRAIN obj-val = 0.78248519
   17-Nov 23:28:01:INFO:root:003800 :: TRAIN obj-val = 0.75499558
   17-Nov 23:30:14:INFO:root:004000 :: TRAIN obj-val = 0.77041978
   17-Nov 23:30:45:INFO:root:
   17-Nov 23:30:45:INFO:root:## Performance on Validation Set
   17-Nov 23:30:45:INFO:root:---------------------------------------------------------
   17-Nov 23:30:45:INFO:root:  Accuracy (avg over 10000) = 70.1800%
   17-Nov 23:30:45:INFO:root:---------------------------------------------------------
   17-Nov 23:30:45:INFO:root:
   17-Nov 23:32:59:INFO:root:004200 :: TRAIN obj-val = 0.94838876


.. code-block:: julia

   ENV["OMP_NUM_THREADS"] = 16
   blas_set_num_threads(16)

.. code-block:: text

   17-Nov 22:24:27:INFO:root:002800 :: TRAIN obj-val = 0.85292178
   17-Nov 22:27:50:INFO:root:003000 :: TRAIN obj-val = 0.88822174
   17-Nov 22:28:31:INFO:root:
   17-Nov 22:28:31:INFO:root:## Performance on Validation Set
   17-Nov 22:28:31:INFO:root:---------------------------------------------------------
   17-Nov 22:28:31:INFO:root:  Accuracy (avg over 10000) = 63.3500%
   17-Nov 22:28:31:INFO:root:---------------------------------------------------------
   17-Nov 22:28:31:INFO:root:
   17-Nov 22:31:58:INFO:root:003200 :: TRAIN obj-val = 1.06246507
   17-Nov 22:35:22:INFO:root:003400 :: TRAIN obj-val = 0.94288993
   17-Nov 22:38:46:INFO:root:003600 :: TRAIN obj-val = 0.84770185
   17-Nov 22:42:12:INFO:root:003800 :: TRAIN obj-val = 0.74366856
   17-Nov 22:45:33:INFO:root:004000 :: TRAIN obj-val = 0.79406691
   17-Nov 22:46:12:INFO:root:
   17-Nov 22:46:12:INFO:root:## Performance on Validation Set
   17-Nov 22:46:12:INFO:root:---------------------------------------------------------
   17-Nov 22:46:12:INFO:root:  Accuracy (avg over 10000) = 67.5700%
   17-Nov 22:46:12:INFO:root:---------------------------------------------------------
   17-Nov 22:46:12:INFO:root:
   17-Nov 22:49:35:INFO:root:004200 :: TRAIN obj-val = 1.02186918

.. code-block:: text

   I1117 21:55:18.451865 33463 solver.cpp:403] Iteration 2800, lr = 0.001
   I1117 21:57:18.176666 33463 solver.cpp:247] Iteration 3000, Testing net (#0)
   I1117 21:57:47.454730 33463 solver.cpp:298]     Test net output #0: accuracy = 0.5853
   I1117 21:57:47.454778 33463 solver.cpp:298]     Test net output #1: loss = 1.1544 (* 1 = 1.1544 loss)
   I1117 21:57:48.058338 33463 solver.cpp:191] Iteration 3000, loss = 1.30168
   I1117 21:57:48.058384 33463 solver.cpp:206]     Train net output #0: loss = 1.30168 (* 1 = 1.30168 loss)
   I1117 21:57:48.058395 33463 solver.cpp:403] Iteration 3000, lr = 0.001
   I1117 21:59:48.495744 33463 solver.cpp:191] Iteration 3200, loss = 1.10434
   I1117 21:59:48.495982 33463 solver.cpp:206]     Train net output #0: loss = 1.10434 (* 1 = 1.10434 loss)
   I1117 21:59:48.495995 33463 solver.cpp:403] Iteration 3200, lr = 0.001
   I1117 22:01:48.953501 33463 solver.cpp:191] Iteration 3400, loss = 1.04567
   I1117 22:01:48.953748 33463 solver.cpp:206]     Train net output #0: loss = 1.04567 (* 1 = 1.04567 loss)
   I1117 22:01:48.953762 33463 solver.cpp:403] Iteration 3400, lr = 0.001
   I1117 22:03:49.428063 33463 solver.cpp:191] Iteration 3600, loss = 1.24852
   I1117 22:03:49.428390 33463 solver.cpp:206]     Train net output #0: loss = 1.24852 (* 1 = 1.24852 loss)
   I1117 22:03:49.428403 33463 solver.cpp:403] Iteration 3600, lr = 0.001
   I1117 22:05:49.946528 33463 solver.cpp:191] Iteration 3800, loss = 0.937274
   I1117 22:05:49.946780 33463 solver.cpp:206]     Train net output #0: loss = 0.937274 (* 1 = 0.937274 loss)
   I1117 22:05:49.946794 33463 solver.cpp:403] Iteration 3800, lr = 0.001
   I1117 22:07:49.897718 33463 solver.cpp:247] Iteration 4000, Testing net (#0)
   I1117 22:08:19.291095 33463 solver.cpp:298]     Test net output #0: accuracy = 0.6098
   I1117 22:08:19.291141 33463 solver.cpp:298]     Test net output #1: loss = 1.09563 (* 1 = 1.09563 loss)
   I1117 22:08:19.894783 33463 solver.cpp:191] Iteration 4000, loss = 1.22756
   I1117 22:08:19.894830 33463 solver.cpp:206]     Train net output #0: loss = 1.22756 (* 1 = 1.22756 loss)
   I1117 22:08:19.894841 33463 solver.cpp:403] Iteration 4000, lr = 0.001
   I1117 22:10:20.511523 33463 solver.cpp:191] Iteration 4200, loss = 1.00094
   I1117 22:10:20.511780 33463 solver.cpp:206]     Train net output #0: loss = 1.00094 (* 1 = 1.00094 loss)
   I1117 22:10:20.511791 33463 solver.cpp:403] Iteration 4200, lr = 0.001


