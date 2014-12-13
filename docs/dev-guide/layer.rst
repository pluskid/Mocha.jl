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

A subtype of ``LayerState`` should be defined correspondingly. For example

.. code-block:: julia

   type PoolingLayerState <: LayerState
     layer      :: PoolingLayer
     blobs      :: Vector{Blob}
     blobs_diff :: Vector{Blob}

     etc        :: Any
   end

A layer state should have a field ``layer`` referencing to the corresponding
``Layer`` object. Other fields and/or behaviors are required depending on the
layer type (see below).

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
