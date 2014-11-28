# News for Mocha Development

## TODO

* Interface
  * Network architecture visualization
* Infrastructure
  * Other solvers?
  * Distributed / Parallel training?
  * CUDA Stream?
* Document
  * Developer's Guide

## v0.0.3 2014.11.27

* Interface
  * IJulia-notebook example
  * Image classifier wrapper
* Network
  * Data transformers for data layers
  * Argmax, Crop, Reshape, HDF5 Output, Weighted Softmax-loss Layers
* Infrastructure
  * Unit tests are extended to cover all layers in both Float32 and Float64
  * Compatibility with Julia v0.3.3 and v0.4 nightly build
* Documentation
  * Complete User's Guide
  * Tutorial on image classification with pre-trained imagenet model

## v0.0.2 2014.11.20

* Infrastructure
  * Ability to import caffe trained model
  * Properly release all the allocated resources upon backend shutdown
* Network
  * Sigmoid activation function
  * Power, Split, Element-wise layers
  * Local Response Normalization layer
  * Channel Pooling layer
  * Dropout Layer
* Documentation
  * Complete MNIST demo
  * Complete CIFAR-10 demo
  * Major part of User's Guide

## v0.0.1 2014.11.13

* Backend
  * Pure Julia CPU
  * Julia + C++ Extension CPU
  * CUDA + cuDNN GPU
* Infrastructure
  * Evaluate on validation set during training
  * Automaticly saving and recovering from snapshots
* Network
  * Convolution layer, mean and max pooling layer, fully connected layer, softmax loss layer
  * ReLU activation function
  * L2 Regularization
* Solver
  * SGD with momentum
* Documentation
  * Demo code of LeNet on MNIST
  * Tutorial document on the MNIST demo (half finished)
