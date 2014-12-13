Layer
=====

A layer in Mocha is an isolate computation component that (optionally) takes some input blobs
and (optionally) produces some output blobs. See :doc:`/user-guide/network` for
an overview of the abstraction of layer and network in Mocha. Implementing
a layer in Mocha means

1. Characterize the layer (e.g. does this layer define a loss function?) so that
   the network topology engine know how to properly glue the layers together to
   build a network.
2. Implementing the computation of the layer, either in a backend-independent
   way, or implement separately for each backend.

Defining a Layer
----------------

A layer, like many other computational components in Mocha, consists of two
parts:

* A layer configuration, a subtype of ``Layer``.
* A layer state, a subtype of ``LayerState``.

``Layer`` defines how a layer should be constructed and should behave, while
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

``@defstruct`` could be used to define a general immutable struct. The first
parameter is the struct name, the second parameter is the super-type and then
a list of struct fields follows. Each field requires a name, a type and
a default value. Optionally, an expression could be added to verify the
user-supplied value meets the requirements.

This macro will automatically define a constructor with keyword arguments for
each field. This makes the interface easier to use for the end-user.

Each layer should have a field ``name``. When the layer produce output blobs, it
should have a property ``tops``, allowing the user to specify a list of names
for the output blobs the layer is producing. If the layer takes any number of
blobs as input, it should also have a property ``bottoms`` for the user to
specify the names for the input blobs. Mocha will use the information specified
in ``tops`` and ``bottoms`` to wire the blobs in a proper data path for network
forward and backward iterations.

A subtype of ``LayerState`` should be defined for each layer correspondingly.
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
written into ``blobs_diff`` by upper layer, and the layer could make use of this
when computing *backward* iteration for itself.

Other fields and/or behaviors are required depending on the layer type (see
below).

Characterizing a Layer
----------------------

Layer is characterized by applying the macro ``@characterize_layer`` to the
defined subtype of ``Layer``. The default characterizations are given by

.. code-block:: julia

   @characterize_layer(Layer,
     is_source  => false, # data layer, takes no bottom blobs
     is_sink    => false, # top layer, produces no top blobs (loss, accuracy, etc.)
     has_param  => false, # contains trainable parameters
     has_neuron => false, # has a neuron
     can_do_bp  => false, # can do back-propagation
     is_inplace => false, # do inplace computation, does not has own top blobs
     has_loss   => false, # produce a loss
     has_stats  => false, # produce statistics
   )

Characterizing a layer could be omitted if all the behaviors are consists with
the default specifications.

Layer Computation API
---------------------

The life cycle of a layer is

1. The user define a ``Layer``
2. The user use defined ``Layer``\ s to construct a ``Net``. The ``Net`` will
   call ``setup`` on each ``Layer`` to construct the corresponding
   ``LayerState``.
3. During training, the solver use a loop to call ``forward`` and ``backward``
   of the ``Net``. The ``Net`` will then call ``forward`` and ``backward`` of
   each layer in a proper order.
4. The user destroy the ``Net``, which will call the ``shutdown`` function of
   each layer.

.. function:: setup_layer(backend, layer, inputs, diffs)

   Construct a corresponding ``LayerState`` object given a ``Layer`` object.
   ``inputs`` is a list of blobs, corresponding to the blobs specified by the
   ``bottoms`` property of the ``Layer`` object. If the ``Layer`` does not have
   ``bottoms`` property, then it will be an empty list.

   ``diffs`` is a list of blobs. Each blob in ``diffs`` corresponds to a blob in
   ``inputs``. When computing back propagation, the back-propagated gradients
   for each input blob should be written into the corresponding one in
   ``diffs``. Blobs in ``inputs`` and ``diffs`` are taken from ``blobs`` and
   ``blobs_diff`` of ``LayerState`` objects of lower layers.

   ``diffs`` is guaranteed to be a list of blobs of the same length
   as ``inputs``. However, when some input blobs does not need back-propagated
   gradients, the corresponding blob in ``diffs`` will be a :class:`NullBlob`.

   This function should setup its own ``blobs`` and ``blobs_diffs`` (if any) by
   possibly measuring the shape of input blobs.

.. function:: forward(backend, layer_state, inputs)

   Do forward computing. It is guaranteed that the blobs in ``inputs`` are
   already computed properly by lower layers. The output blobs (if any) should
   be written into the blobs in the ``blobs`` field of the layer state.

.. function:: backward(backend, layer_state, inputs, diffs)

   Do backward computing. It is guaranteed that the back-propagated gradients
   with respect to all the output blobs for this layer are already computed
   properly and written into the blobs in the ``blobs_diff`` field of the layer
   state. This function should compute the gradients with respect to its
   parameters (if any). It is also responsible to compute the back-propagated
   gradients and write into the blobs in ``diffs``. If a blob in ``diffs`` is
   a :class:`NullBlob`, computation for the back-propagated gradients for that
   blob could be omitted.

   The contents in the blobs in ``inputs`` are the same as in the last call of
   ``forward``, and could be used if necessary.

   If a layer does not do backward propagation (e.g. a data layer), an empty
   ``backward`` function should still be defined explicitly.

.. function:: shutdown(backend, layer_state)

   Release all the resources allocated in ``setup``.
