Layer
=====

A layer in Mocha is an isolated computation component that (optionally) takes some input blobs
and (optionally) produces some output blobs. See :doc:`/user-guide/network` for
an overview of the abstraction of *layer* and *network* in Mocha. Implementing
a layer in Mocha means

1. Characterizing the layer (e.g. does this layer define a loss function?) so that
   the network topology engine knows how to properly glue the layers together to
   build a network.
2. Implementing the computation of the layer, either in a backend-independent
   way, or separately for each backend.

Defining a Layer
----------------

A layer, like many other computational components in Mocha, consists of two
parts:

* A layer configuration, a subtype of ``Layer``.
* A layer state, a subtype of ``LayerState``.

``Layer`` defines how a layer should be constructed and it should behave, while
``LayerState`` is the realization of a layer which actually holds the data
blobs.

Mocha has a helper macro ``@defstruct`` to define a ``Layer`` subtype. For
example

.. code-block:: julia

   @defstruct PoolingLayer Layer (
     name :: String = "pooling",
     (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) > 0),
     (tops :: Vector{Symbol} = Symbol[], length(tops) == length(bottoms)),
     (kernel :: NTuple{2, Int} = (1,1), all([kernel...] .> 0)),
     (stride :: NTuple{2, Int} = (1,1), all([stride...] .> 0)),
     (pad :: NTuple{2, Int} = (0,0), all([pad...] .>= 0)),
     pooling :: PoolingFunction = Pooling.Max(),
     neuron :: ActivationFunction = Neurons.Identity(),
   )

``@defstruct`` can be used to define a general immutable struct. The first
parameter is the struct name, the second parameter is the super-type and then
a list of struct fields follows. Each field requires a name, a type and
a default value. Optionally, an expression can be added to verify the
user-supplied value meets the requirements.

This macro will automatically define a constructor with keyword arguments for
each field. This makes the interface easier to use for the end-user.

Each layer needs to have a field ``name``. When the layer produce output blobs, it
has to have a property ``tops``, allowing the user to specify a list of names
for the output blobs the layer is producing. If the layer takes any number of
blobs as input, it should also have a property ``bottoms`` for the user to
specify the names for the input blobs. Mocha will use the information specified
in ``tops`` and ``bottoms`` to wire the blobs in a proper data path for network
forward and backward iterations.

A subtype of ``LayerState`` should be defined for each layer, correspondingly.
For example

.. code-block:: julia

   type PoolingLayerState <: LayerState
     layer      :: PoolingLayer
     blobs      :: Vector{Blob}
     blobs_diff :: Vector{Blob}

     etc        :: Any
   end

A layer state should have a field ``layer`` referencing to the corresponding
``Layer`` object. If the layer produce output blobs, the state should have
a field called ``blobs``, and the layer will write output into ``blobs`` during
each *forward* iteration. If the layer needs back-propagation from the upper
layers, the state should also have a field called ``blobs_diff``. Mocha will
pass the blobs in ``blobs_diff`` to the function computing *backward* iteration
in the corresponding upper layer. The back-propagated gradients will be
written into ``blobs_diff`` by the upper layer, and the layer can make use of this
when computing the *backward* iteration.

Other fields and/or behaviors are required depending on the layer type (see
below).

Characterizing a Layer
----------------------

A layer is characterized by applying the macro ``@characterize_layer`` to the
defined subtype of ``Layer``. The default characterizations are given by

.. code-block:: julia

   @characterize_layer(Layer,
     is_source  => false, # data layer, takes no bottom blobs
     is_sink    => false, # top layer, produces no top blobs (loss, accuracy, etc.)
     has_param  => false, # contains trainable parameters
     has_neuron => false, # has a neuron
     can_do_bp  => false, # can do back-propagation
     is_inplace => false, # does inplace computation, does not have own top blobs
     has_loss   => false, # produces a loss
     has_stats  => false, # produces statistics
   )

Characterizing a layer can be omitted if all the behaviors are consists with
the default specifications. The characterizations should be self-explanatory by
the name and comments above. Some characterizations come with extra
requirements:

``is_source``
  The layer will be used as a source layer of a network. Thus it should take no
  input blob and the ``Layer`` object should have no ``bottoms`` property.
``is_sink``
  The layer will be used as a sink layer of a network. Thus it should produce no
  output blob, and the ``Layer`` object should have no ``tops`` property.
``has_param``
  The layer has trainable parameters. The ``LayerState`` object should have
  a ``parameters`` field, containing a list of :class:`Parameter` objects.
``has_neuron``
  The ``Layer`` object should have a property called ``neuron`` of type
  :class:`ActivationFunction`.
``can_do_bp``
  Should be true if the layer has the ability to do back propagation.
``is_inplace``
  An inplace ``Layer`` object should have no ``tops`` property because the
  output blobs are the same as the input blobs.
``has_loss``
  The ``LayerState`` object should have a ``loss`` field.
``has_stats``
  The layer computes statistics (e.g. accuracy). The statistics should be
  accumulated across multiple mini-batches, until the user explicit reset the
  statistics. The following functions should be implemented for the layer

  .. function:: dump_statistics(storage, layer_state, show)

     ``storage`` is a data storage (typically a :class:`CoffeeLounge` object)
     that is used to dump statistics into, via the function
     ``update_statistics(storage, key, value)``.

     ``show`` is a boolean value, when true, indicating that a summary of the
     statistics should also be printed to stdout.

  .. function:: reset_statistics(layer_state)

     Reset the statistics.


Layer Computation API
---------------------

The life cycle of a layer is

1. The user defines a ``Layer``
2. The user uses ``Layer``\ s to construct a ``Net``. The ``Net`` will
   call ``setup_layer`` on each ``Layer`` to construct the corresponding
   ``LayerState``.
3. During training, the solver use a loop to call the ``forward`` and ``backward``
   functions of the ``Net``. The ``Net`` will then call ``forward`` and ``backward`` of
   each layer in a proper order.
4. The user destroys the ``Net``, which will call the ``shutdown`` function of
   each layer.

.. function:: setup_layer(backend, layer, inputs, diffs)

   Construct a corresponding ``LayerState`` object given a ``Layer`` object.
   ``inputs`` is a list of blobs, corresponding to the blobs specified by the
   ``bottoms`` property of the ``Layer`` object. If the ``Layer`` does not have
   a ``bottoms`` property, then it will be an empty list.

   ``diffs`` is a list of blobs. Each blob in ``diffs`` corresponds to a blob in
   ``inputs``. When computing back propagation, the back-propagated gradients
   for each input blob should be written into the corresponding one in
   ``diffs``. Blobs in ``inputs`` and ``diffs`` are taken from ``blobs`` and
   ``blobs_diff`` of the ``LayerState`` objects of lower layers.

   ``diffs`` is guaranteed to be a list of blobs of the same length
   as ``inputs``. However, when some input blobs do not need back-propagated
   gradients, the corresponding blob in ``diffs`` will be a :class:`NullBlob`.

   This function should set up its own ``blobs`` and ``blobs_diffs`` (if any),
   matching the shape of its input blobs.

.. function:: forward(backend, layer_state, inputs)

   Do forward computing. It is guaranteed that the blobs in ``inputs`` are
   already computed by the lower layers. The output blobs (if any) should
   be written into the blobs in the ``blobs`` field of the layer state.

.. function:: backward(backend, layer_state, inputs, diffs)

   Do backward computing. It is guaranteed that the back-propagated gradients
   with respect to all the output blobs for this layer are already computed
   and written into the blobs in the ``blobs_diff`` field of the layer
   state. This function should compute the gradients with respect to its
   parameters (if any). It is also responsible to compute the back-propagated
   gradients and write them into the blobs in ``diffs``. If a blob in ``diffs`` is
   a :class:`NullBlob`, computation for the back-propagated gradients for that
   blob can be omitted.

   The contents in the blobs in ``inputs`` are the same as in the last call of
   ``forward``, and can be used if necessary.

   If a layer does not do backward propagation (e.g. a data layer), an empty
   ``backward`` function still has to be defined explicitly.

.. function:: shutdown(backend, layer_state)

   Release all the resources allocated in ``setup_layer``.

Layer Parameters
----------------

If a layer has train-able parameters, it should define a ``parameters`` field in
the ``LayerState`` object, containing a list of :class:`Parameter` objects. It
should also define the ``has_param`` characterization. The only computation
the layer needs to do, is to compute the gradients with respect to each
parameter and write them into the ``gradient`` field of each :class:`Parameter`
object.

Mocha will handle the updating of parameters during training automatically.
Other parameter-related issues like initialization, regularization and norm
constraints will also be handled automatically.

Layer Activation Function
-------------------------

When it makes sense for a layer to have an activation function, it can add
a ``neuron`` property to the ``Layer`` object and define the ``has_neuron``
characterization. Everything else will be handled automatically.

