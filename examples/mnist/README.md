The two scripts in this folder can be used to train example models on MNIST.

###mnist.jl 
This trains a LeNet like convolutional neural network on the MNIST dataset (see [the LeNet paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) for a description of the model).

###mnist_dropout_fc.jl 
This trains a fully connected two layer neural network on MNIST and reproduces the results of the [original dropout paper](http://arxiv.org/abs/1207.0580).
It should currently bottom out at 99.05 % accuracy (or 0.95 % error) on the test set.

###mnist_learning_curve.jl
This plots the learning curve from the data saved in the snapshots directory and can be run while the network is traiing to verify that the solver is converging.

NOTE: these scripts currently do not select learning parameters using a validation set and hence should be taken with a grain of salt.
