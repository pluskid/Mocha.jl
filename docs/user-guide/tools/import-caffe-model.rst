Importing Trained Model from Caffe
==================================

Overview
--------

Mocha provides a tool to help with importing Caffe's trained models. Importing
Caffe's models consists of two steps:

#. **Translating the network architecture definitions**: this needs to be done
   manually. Typically for each layer used in Caffe, there is an equivalent
   in Mocha, so translating should be relatively straightforward. See :doc:`the
   CIFAR-10 tutorial </tutorial/cifar10>` for an example of translating Caffe's
   network definition. You need to make sure to use the same name for the layers so
   that when importing the learned parameters, Mocha is able to find the
   correspondences.
#. **Importing the learned network parameters**: this can be done
   automatically, and is the main topic of this document.

Caffe uses a binary protocol buffer file to store trained models. Instead of
parsing this complicated binary file, we provide a tool to export the model
parameters to the HDF5 format, and import the HDF5 file from Mocha. As
a result, you need to have Caffe installed to do the importing.

Exporting Caffe's Snapshot to HDF5
----------------------------------

Caffe's snapshot files contain some extra information, but what we need are only the
learned network parameters. The strategy is to use Caffe's built-in API to load
their model snapshot, and then iterate all network layers in memory to dump the
layer parameters to a HDF5 file. In the ``tools`` directory of Mocha's source
root, you can find this in ``dump_network_hdf5.cpp``.

Put that file in Caffe's ``tools`` directory, and re-compile Caffe. The tool
should be built automatically, and the executable file could typically be found
in ``build/tools/dump_network_hdf5``. Run the tool as follows:

.. code-block:: bash

   build/tools/dump_network_hdf5 \
       examples/cifar10/cifar10_full_train_test.prototxt \
       examples/cifar10/cifar10_full_iter_70000.caffemodel \
       cifar10.hdf5

where the arguments are Caffe's network definition, Caffe's model snapshot you
want to export and the output HDF5 file, respectively.

Currently, in all the :doc:`layers Mocha supports </user-guide/layers/index>`,
only :class:`InnerProductLayer` and :class:`ConvolutionLayer` contains trained
parameters. When some other layers are needed, it should be straightforward to
modify ``dump_network_hdf5.cpp`` to include proper rules for exporting.

Importing the HDF5 Snapshot to Mocha
--------------------------------

Mocha has a unified interface to import the HDF5 model we just exported. After
constructing the network with the same architecture as translated from Caffe,
you can import the HDF5 file by calling

.. code-block:: julia

   using HDF5
   h5open("/path/to/cifar10.hdf5", "r") do h5
     load_network(h5, net)
   end

Actually, ``net`` does not need to be the exactly the same architecture. What it
does is to try to find the parameters for each layer in the HDF5 archive. So if
the Mocha architecture contains fewer layers, it should be fine.

By default, if the parameters for a layer can not be found in the HDF5
archive, it will fail with an error. But you can also change this behavior by
passing ``false`` as the third argument, indicating not to panic if parameters
are not found in the archive. In this case, Mocha will use the associated
:doc:`initializer </user-guide/initializer>` to initialize the parameters not
found in the archive.

Mocha's HDF5 Snapshot Format
----------------------------

By using the same technique, you can import network parameters trained by other
deep learning tools into Mocha, as long as you can export them to HDF5 files. The
HDF5 file that Mocha tries to import is very simple

* Each parameter (e.g. the filter of a convolution layer) is stored as a 4D
  tensor dataset in the HDF5 file.
* The dataset name for each parameter should be ``layer___param``. For example,
  ``conv1___filter`` is for the ``filter`` parameter of the convolution layer
  with the name ``conv1``.

  The HDF5 file format supports hierarchy. But it is rather complicated to
  manipulate hierarchies in some tools (e.g. the `HDF5 Lite
  <http://www.hdfgroup.org/HDF5/doc/HL/RM_H5LT.html>`_ library Caffe is using),
  so we decided to use a simple flat format.
* In Caffe, the ``bias`` parameter for a convolution layer and an inner product
  layer is optional. It is OK to omit them on exporting if there is no bias. You
  will get a warning message when importing in Mocha. Mocha will use the
  associated initializer (by default initializing to 0) to initialize the bias.

Exporting Caffe's Mean File
------------------------

Sometimes Caffe's model includes a *mean file*, which is the mean data point
computed over all the training data. This information might be needed in :doc:`data
preprocessing </user-guide/data-transformer>`. Of course we could compute the
mean from the training data manually. But if the training data is too large or is
not easily obtainable, it might be easier to load Caffe's pre-computed mean file
instead.

In the ``tools`` directory of Mocha's source root, you can find
``dump_mean_file.cpp``. Similar to exporting Caffe's model file, you can copy
this file to Caffe's ``tools`` directory and compile Caffe. After that, you can export
Caffe's mean file:

.. code-block:: bash

   build/tools/dump_mean_file \
       data/ilsvrc12/imagenet_mean.binaryproto \
       ilsvr12_mean.hdf5

The exported HDF5 file can then be loaded in Mocha using :class:`DataTransformers.SubMean`.


