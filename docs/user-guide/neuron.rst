Neurons (Activation Functions)
==============================

They could be attached to any layers. The neuron of each layer will affect the
output in the forward pass and the gradient in the backward pass automatically
unless it is an identity neuron. A layer have an identity neuron by default [1]_.

.. [1] This is actually not true: not all layers in Mocha support neurons. For
   example, data layers currently does not have neurons, but this feature could
   be added by simply adding a neuron property to the data layer type. However,
   for some layer types like loss layers or accuracy layers, it does not make
   much sense to have neurons.
