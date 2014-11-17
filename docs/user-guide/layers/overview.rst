Overview
~~~~~~~~

There are four basic layer types in Mocha:

Data Layers
  Read data from source and feed them to top layers.
Computation Layer
  Take input stream from bottom layers, carry out computations and feed the
  computed results to top layers.
Loss Layers
  Take computed results (and ground truth labels) from bottom layers, compute
  a real number loss. Loss values from all the loss layers and regularizers in
  a net are added together to define the final loss function of the net. The
  loss function will be used to train the net parameters in back propagation.
Statistics Layers
  Take input from bottom layers and compute useful statistics like
  classification accuracy. Statistics could be accumulated throughout multiple
  iterations.

Notations
---------

In the document for the layers, the following common notations will be used

* :math:`x`: the input tensor of a layer.
* :math:`y`: the output tensor of a layer.
