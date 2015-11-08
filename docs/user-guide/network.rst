Networks
========

Overview
--------

In deep learning, computations are abstracted into relatively isolated *layers*.
The layers are connected together according to a given *architecture* that
describes a data flow. Starting with the data layer: it takes input from
a dataset or user input, does some data pre-processing, and then produces a stream
of processed data. The output of the data layer is connected to the input of some computation
layer, which again produces a stream of computed output that gets connected to
the input of some upper layers. At the top of a network, there is typically
a layer that produces the network prediction or computes the loss function value
according to provided ground-truth labels.

During training, the same data path, except in the reversed direction, is used
to propagate the error back to each layer using chain rules. Via back
propagation, each layer can compute the gradients for its own parameters, and
update the parameters according to some optimization schemes. Again, the
computation is abstracted into layers.

The abstraction and separation of *layers* from the *architecture* is important. The
library implementation can focus on each layer type independently, and does
not need to worry about how those layers are going to be connected with each
other. On the other hand, the network designer can focus on the architecture,
and does not need to worry about the internal computations of layers. This
enables us to compose layers almost arbitrarily to create very deep
/ complicated networks. The network could be carrying out highly sophisticated
computations when viewed as a whole, yet all the complexities are nicely
decomposed into manageable pieces.

Most of the illustrations for (deep) neural networks look like the following image
stolen from `Wikipedia's page on Artificial Neural Networks
<https://en.wikipedia.org/wiki/Artificial_neural_network>`_:

.. image:: images/ANN.*

When writing Mocha, I found this kind of illustrations a bit confusing, as it
does not align well with the abstract concept of *layers* we just described. In
our abstraction, the computation is done **within** each layer, and the network
architecture specifies the data path connections for the layers only.
In the figure above, the "Input", "Hidden", and "Output" labels are put on the
nodes, suggesting the nodes are layers. However, the nodes do not
computate anything, instead, computations are specified by the arrows connecting these
nodes.

I think the following kind of illustration is clearer, for
the purpose of abstracting *layers* and *architectures* separately:

.. image:: images/NN-view.*

Each layer is now represented as a *box* that has inputs (denoted by :math:`x^L`
for the :math:`L`-th layer) and outputs (denoted by :math:`y^L`). Now the
architecture specifies which layer's outputs connect to which layer's inputs
(the dark lines in the figure). On the other hand, the intra-layer connections,
or computations (see dotted line in the figure), should be isolated from the
outside world.

.. note::

   Unlike the intra-layer connections, the inter-layer connections are drawn as
   simple parallel lines, because they are essentially a point-wise copying
   operation. Because all the computations are abstracted to be inside the
   layers, there is no real computation in between them. Mathematically, this
   means :math:`x^L=y^{L-1}`. In actual implementation, data copying is avoided
   via data sharing.

Of course, the choice is only a matter of taste, but as we will see, using the
latter kind of illustration makes it much easier to understand Mocha's internal
structure and end-user interface.

Network Architecture
--------------------

Specifying a network architecture in Mocha means defining a set of layers, and
connecting them. Taking the figure above for example, we could define a data
layer and an inner product layer

.. code-block:: julia

   data_layer = HDF5DataLayer(name="data", source="data-list.txt", batch_size=64, tops=[:data])
   ip_layer   = InnerProductLayer(name="ip", output_dim=500, tops=[:ip], bottoms=[:data])

Note how the ``tops`` and ``bottoms`` properties give names to the output and input
of the layer. Since the name for the input of ``ip_layer`` matches the name for
the output of ``data_layer``, they will be connected as shown in the figure
above. The softmax layer could be defined similarly. Mocha will do a topological
sort on the collection of layers and automatically figure out the connection
defined implicitly by the names of the inputs and outputs of each layer.

Layer Implementation
--------------------

The layer is completely unaware of what happens in the outside world. Two
important procedures need to be defined to implement a layer:

* Feed-forward: given the inputs, compute the outputs. For example, for the
  inner product layer, it will compute the outputs as :math:`y_i = \sum_j
  w_{ij}x_j`.
* Back-propagate: given the errors propagated from upper layers, compute the
  gradient of the layer parameters, **and** propagate the error
  down to lower layers. Note this is described in very vague terms like
  *errors*. Depending on the abstraction we choose here, these vague terms become
  a concrete meaning.

Specifically, back-propagation is used during network training, when an
optimization algorithm wants to compute the gradient of each parameter with
respect to an *objective function*. Typically, the objective function is some
loss function that penalizes incorrect predictions given the ground-truth labels.
Let's call the objective function :math:`\ell`.

Now let's switch to the viewpoint of an inner product layer: it needs to compute
the gradients of the weights parameters :math:`w` with respect to :math:`\ell`.
Of course, since we restrict the layer from accessing the outside world, it does
not know what :math:`\ell` is. But the gradients could be computed via chain
rule

.. math::

   \frac{\partial \ell}{\partial w_{ij}} = {\color{red}{\frac{\partial y_i}{\partial
   w_{ij}}}}\times {\color{blue}{\frac{\partial \ell}{\partial y_i}}}

The red part can be computed **within** the layer, and the blue part are the
so-called "errors propagated from the upper layers". It comes from the reversed
data path as used in the feed-forward pass.

Now our inner product layer is ready to "propagate the errors down to lower
layers", precisely speaking, this means computing

.. math::

   \frac{\partial \ell}{\partial x_i} = \sum_j {\color{red}{\frac{\partial
   y_j}{\partial x_i}}}\times{\color{blue}{\frac{\partial \ell}{\partial y_j}}}

Again, this is decomposed into a part that can be computed internally and
a part that comes from the "top". Recall we said the :math:`L`-th layer's inputs
:math:`x^L_i` are equal to the :math:`(L-1)`-th layer's outputs
:math:`y^{L-1}_i`. That means what we just computed

.. math::

   \frac{\partial \ell}{\partial x^L_i} = \frac{\partial \ell}{\partial y^{L-1}_i}

is exactly what the lower layer's "errors propagated from upper layers". By
tracing the whole data path reversely, we now help each layers compute the
gradients of their own parameters internally. And this is called
back-propagation.

Mocha Network Topology Tips
---------------------------

Shared Parameters
~~~~~~~~~~~~~~~~~

Consider a case where you want to construct *two* (or more) networks that share parameters. For
example, during training, you want to have a *validation net* that shares
parameters with the *training net*, yet takes a different data layer as input
data stream and computes the accuracy on the validation set. In this case,
simply using *the same* ``Layer`` object when constructing both networks will be
enough. See :doc:`/tutorial/mnist` for a concrete example.

If you want to have **different** layers in **the same** network to share
parameters, you can just use the same ``param_key`` property in the layers you
want to share parameters. For example

.. code-block:: julia

   layer_ip1 = InnerProductLayer(name="ip1", param_key="shared_ip",
       output_dim=512, bottoms=[:input1], tops=[:output1])
   layer_ip2 = InnerProductLayer(name="ip2", param_key="shared_ip",
       output_dim=512, bottoms=[:input2], tops=[:output2])

If the two (or more) layers sharing parameters are of the same type (this is
almost always true), an easier and more efficient way to do the same thing is
simply to define one layer that takes multiple inputs and produce multiple
outputs. For example, the snippet above is equivalent to

.. code-block:: julia

   layer_ip = InnerProductLayer(name="ip", output_dim=512,
       bottoms=[:input1,:input2], tops=[:outpu1,:outpu2])

Not all layers accept multiple input blobs. Some layers require all the
input blobs to be the same shape, while others can handle input blobs of
completely different shapes. Please refer to the ``bottoms`` and ``tops``
properties of each layer for the detailed behavior of each layer.

Shared Blobs
~~~~~~~~~~~~

In the basic case, a data path connects each output blob to one input blob. In
some cases, one output could be used in multiple places. For example, in a test
net, the output of the top representation layer will be used to compute the
predictions, and produce either loss or accuracy; meanwhile, one might want to
use a :class:`HDF5OutputLayer` to store the representations as extracted
features for future use. When the network is only doing *forward* operation,
blob sharing is not a problem: multiple layers could be declared to take the
same blob as input.

When you want to do *backward* operation (i.e. back-propagation for training) on
the network, things could get a little bit complicated: If back-propagation does
not go through the blob, than sharing is OK. For example, the output blob of
a :class:`HDF5DataLayer` does not need back-propagation. The output blob of
a :class:`ReshapeLayer` sitting directly on top of a data layer does not need
back-propagation, either.

However, for a :class:`InnerProductLayer`, even sitting directly on top of
a data layer, its output blobs do need back-propagation, because the inner
product layer needs back-propagation to compute gradients with respect to its
weights and bias parameters. A :class:`TopologyError` will be thrown when you
try to do back-propagation on a network with this kind of Topology.

In this case, a :class:`SplitLayer` can be used to explicitly "split" a blob
into two (or more) "copies". The split layer can handle back-propagation
correctly. Moreover, the forward operation of a split layer is implemented with
data sharing instead of copying. Thus no extra cost is incurred during the forward
pass.

Debugging
---------

Mocha provides some utilities to show the structure of a network, which might be
useful for debugging. First of all, you can just call ``println`` on a network
object, the sorted layers will be printed, with basic information including blob
names and shapes, etc. Alternatively, one can call ``net2dot`` to dump the
network structure to a dot file, a script used by `GraphViz
<http://www.graphviz.org/>`_. For example, if you have GraphViz installed, the
following command

.. code-block:: julia

   open("net.dot", "w") do out net2dot(out, net) end
   run(`dot -Tpng net.dot` |> "net.png")

will generate a visualization of the network architecture in ``net.png``. The
following is a visualization of the network used in the :doc:`MNIST
</tutorial/mnist>` example.

.. image:: images/mnist-dot.*

