# News for Mocha Development

## v0.0.9 2015.07.20

* Infrastructure
  * Add JLD.jl in REQUIREMENT as it becomes an independent package
  * Fix a Julia v0.4-dev compatability
* Interface
  * GraphViz visualization for network architecture

## v0.0.8 2015.05.27

* Interface
  * Option to display other information for training summary (@bisraelsen)
* Infrastructure
  * Improved gradient check (@steven-varga)
  * Fix temp file issue for unit-test on Windows
  * Fix XavierInitializer scale (@adambrewster)
  * Option to specify GPU device
* Network
  * Index2Onehot layer, MemoryOutputLayer
  * SoftmaxLayer now can do backward

## v0.0.7 2015.02.27

* Infrastructure
  * Boltzmann.jl now supports DBN pre-training for Mocha.jl
  * Clearer Nesterov solver (@the-moliver)
  * Staged momentum policy
  * Learning rate policy to decay dynamically based on performance on validation set
* Network
  * Async HDF5 data layer: faster and with chunking to support fast data shuffling
  * Softlabel-softmax-loss layer allows training with posterior (instead of hard labels) as labels
  * Weight loss layers to combine multiple loss functions in one network
  * Square loss layer is now capable of propagating gradients to both sides

## v0.0.6 2014.12.31

* Infrastructure
  * Numerical gradient checking in unit-tests (@pcmoritz)
  * Simple ref-counting for shared parameters
* Network
  * RandomMaskLayer, TiedInnerProductLayer, IdentityLayer
  * Freezing / unfreezing some layers of a network to allow layer-wise pre-training
* Documentation
  * A new tutorial on MNIST that compares unsupervised pre-training via stacked denoising auto-encoders and random initialization

## v0.0.5 2014.12.20

* Infrastructure
  * **{Breaking Changes}** cuDNN 6.5 R2 (Release Candidate) (@JobJob)
    * cuDNN 6.5 R2 is **NOT** backward compatible with 6.5 R1
    * Forward convolution speed up
    * Pooling with padding is supported
    * Mac OS X is supported
  * 4D-tensor -> ND-tensor
    * Mocha is now capable of handling general ND-tensor
    * Except that (for now) `ConvolutionLayer` and `PoolingLayer` still requires the inputs to be 4D
    * The generalization is *almost* backward compatible, except
      * The interface for `ReshapeLayer` changed b/c the target shape needs to be ND, instead of 4D now
      * Parameters added for some layers to allow the user to specify which dimension to operate on
      * The output of `InnerProductLayer` is now 2D-tensor instead of 4D
    * Unit-tests are expanded to cover test cases for ND-tensor when applicable
* Interface
  * print a constructed `Net` to get a brief overview of the geometry of input/output blobs in each layers
* Documentation
  * Setup the [Roadmap Ticket](https://github.com/pluskid/Mocha.jl/issues/22), duscussions/suggestions are welcome
  * Update everything to reflect 4D -> ND tensor changes
  * Document for parameter norm constraints
  * Developer's Guide for blob and layer API

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
