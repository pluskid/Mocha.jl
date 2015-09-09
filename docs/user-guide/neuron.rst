Neurons (Activation Functions)
==============================

Neurons can be attached to any layer. The neuron of each layer will affect the
output in the forward pass and the gradient in the backward pass automatically
unless it is an identity neuron. Layers have an identity neuron by default [1]_.

.. class:: Neurons.Identity

   An activation function that does not change its input.

.. class:: Neurons.ReLU

   Rectified Linear Unit. During the forward pass, it inhibits all negative
   activations. In other words, it computes point-wise :math:`y=\max(0, x)`. The
   point-wise derivative for ReLU is

   .. math::

      \frac{dy}{dx} = \begin{cases}1 & x > 0 \\ 0 & x \leq 0\end{cases}

   .. note::

      ReLU is actually not differentialble at 0. But it has *subdifferential*
      :math:`[0,1]`. Any value in that interval can be taken as
      a *subderivative*, and can be used in SGD if we generalize from gradient
      descent to *subgradient* descent. In the implementation, we choose the subgradient at :math:`x==0` to be 0.
      
.. class:: Neurons.LReLU

   Leaky Rectified Linear Unit. A Leaky ReLU can help fix the "dying ReLU" problem. ReLU's
   can "die" if a large enough gradient changes the weights such that the neuron never activates
   on new data.
   
   .. math::

      \frac{dy}{dx} = \begin{cases}1 & x > 0 \\ 0.01 & x \leq 0\end{cases}

.. class:: Neurons.Sigmoid

   Sigmoid is a smoothed step function that produces approximate 0 for negative
   input with large absolute values and approximate 1 for large positive inputs.
   The point-wise formula is :math:`y = 1/(1+e^{-x})`. The point-wise derivative
   is

   .. math::

      \frac{dy}{dx} = \frac{-e^{-x}}{\left(1+e^{-x}\right)^2} = (1-y)y

.. class:: Neurons.Tanh

   Tanh is a transformed version of Sigmoid, that takes values in :math:`\pm 1`
   instead of the unit interval.
   input with large absolute values and approximate 1 for large positive inputs.
   The point-wise formula is :math:`y = (1-e^{-2x})/(1+e^{-2x})`. The point-wise
   derivative is

   .. math::

      \frac{dy}{dx} = 4e^{2x}/(e^{2x} + 1)^2 = (1-y^2)

.. [1] This is actually not true: not all layers in Mocha support neurons. For
   example, data layers currently does not have neurons, but this feature could
   be added by simply adding a neuron property to the data layer type. However,
   for some layer types like loss layers or accuracy layers, it does not make
   much sense to have neurons.
