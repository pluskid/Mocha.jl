Initializers
============

Initializers provide init values for network parameter blobs. In Caffe, they are
called *Fillers*.

.. class:: NullInitializer

   An initializer that does nothing. To initialize with zeros, use a ConstantInitializer.

.. class:: ConstantInitializer

   Set everything to a constant.

   .. attribute:: value

      The value used to initialize a parameter blob. Typically this is set to 0.

.. class:: XavierInitializer

   An initializer based on [BengioGlorot2010]_, but does not use the fan-out
   value. It fills the parameter blob by randomly sampling uniform data from
   :math:`[-S,S]` where the scale :math:`S=\sqrt{3 / F_{\text{in}}}`. Here
   :math:`F_{\text{in}}` is the fan-in: the number of input nodes.

   Heuristics are used to determine the fan-in: For a ND tensor parameter blob,
   the product of all the 1 to N-1 dimensions are considered as fan-in, while
   the last dimension is considered as fan-out.

   .. [BengioGlorot2010] Y. Bengio and X. Glorot, *Understanding the
      difficulty of training deep feedforward neural networks*, in Proceedings of
      AISTATS 2010, pp. 249-256.

.. class:: GaussianInitializer

   Initialize each element in the parameter blob as independent and identically
   distributed Gaussian random variables.

   .. attribute:: mean

      Default 0.

   .. attribute:: std

      Default 1.


.. class:: OrthogonalInitializer

   Initialize the parameter blob to be a random orthogonal matrix (i.e. :math:`W^TW=I`),
   times a scalar gain factor.  Based on [Saxe2013]_.

   .. [Saxe2013] Andrew M. Saxe, James L. McClelland, Surya Ganguli, *Exact solutions to
		 the nonlinear dynamics of learning in deep linear neural networks*,
		 http://arxiv.org/abs/1312.6120 with a presentation https://www.youtube.com/watch?v=Ap7atx-Ki3Q

   .. attribute:: gain

      Default 1.  Use :math:`\sqrt{2}` for layers with ReLU activations.
