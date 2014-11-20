# News for Mocha Development

## TODO

* Interface
  * IJulia-notebook example
  * Network architecture visualization
* Infrastructure
  * Other solvers?
  * CUDA Stream?
* Document
  * Complete User's Guide
  * Developer's Guide

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
