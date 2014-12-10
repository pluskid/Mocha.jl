# News for Mocha Development

## TODO

* Interface
  * Network architecture visualization
* Infrastructure
  * CUDA Stream?
* Document
  * Developer's Guide

## v0.0.4 2014.12.09

* Network
  * Parameter (l2-norm) constraints (@stokasto)
  * Random shuffling for HDF5 data layer
  * ConcatLayer
* Infrastructure
  * Momentum policy (@stokasto)
  * Save training statistics to file and plot tools (@stokasto)
  * Coffee breaks now have a coffee lounge
  * Auto detect whether CUDA kernel needs update
  * Stochastic Nesterov Accelerated Gradient Solver
  * Solver refactoring:
    * Behaviors for coffee breaks are simplified
    * Solver state variables like iteration now has clearer semantics
    * Support loading external pre-trained models for fine-tuning
  * Support explicit weight-sharing layers
  * Behaviors of layers taking multiple inputs made clear and unit-tested
  * Refactoring:
    * Removed the confusing `System` type
    * `CuDNNBackend` renamed to `GPUBackend`
    * Cleaned up `cuBLAS` API (@stokasto)
  * Layers are now organized by characterization properties
  * Robustness
    * Various explicit topology verifiecations for `Net` and unit tests
    * Increased unit test coverage for rare cases
  * Updated dependency to HDF5.jl 0.4.7
* Documentation
  * A new MNIST example using fully connected and dropout layers (@stokasto)
  * Reproducible MNIST results with fixed random seed (@stokasto)
  * Tweaked IJulia Notebook image classification example
  * Document for solvers and coffee breaks

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
