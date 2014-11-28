# Mocha

[![Build Status](https://travis-ci.org/pluskid/Mocha.jl.svg?branch=master)](https://travis-ci.org/pluskid/Mocha.jl)
[![Documentation Status](https://readthedocs.org/projects/mochajl/badge/?version=latest)](http://mochajl.readthedocs.org/)
[![Mocha](http://pkg.julialang.org/badges/Mocha_release.svg)](http://pkg.julialang.org/?pkg=Mocha&ver=release)
[![Coverage Status](https://img.shields.io/coveralls/pluskid/Mocha.jl.svg)](https://coveralls.io/r/pluskid/Mocha.jl?branch=master)
<!--[![Build status](https://ci.appveyor.com/api/projects/status/342vcj5lj2jyegsp?svg=true)](https://ci.appveyor.com/project/pluskid/mocha-jl)-->

Mocha is a Deep Learning framework for [Julia](http://julialang.org/), inspired by the C++ Deep Learning framework [Caffe](http://caffe.berkeleyvision.org/). Mocha support multiple backends:

- Pure Julia CPU Backend: Implemented in pure Julia; Runs out of the box without any external dependency; Reasonably fast on small models thanks to Julia's LLVM-based just-in-time (JIT) compiler and [Performance Annotations](http://julia.readthedocs.org/en/latest/manual/performance-tips/#performance-annotations) that eliminate unnecessary bound checkings.
- CPU Backend with Native Extension: Some bottleneck computations (Convolution and Pooling) have C++ implementations. When compiled and enabled, could be faster than the pure Julia backend.
- CUDA + cuDNN: An interface to NVidia® [cuDNN](https://developer.nvidia.com/cuDNN) GPU accelerated deep learning library. When run with CUDA GPU devices, could be much faster depending on the size of the problem (e.g. on MNIST CUDA backend is roughly 20 times faster than the pure Julia backend).

## Installation

To install the release version, simply run

```julia
Pkg.add("Mocha")
```

in Julia console. To install the latest development version, run the following command instead:

```julia
Pkg.clone("https://github.com/pluskid/Mocha.jl.git")
```

Then you can run the built-in unit tests with

```julia
Pkg.test("Mocha")
```

to verify that everything is functioning properly on your machine.

## Hello World

Please refer to [the MNIST tutorial](http://mochajl.readthedocs.org/en/latest/tutorial/mnist.html) on how prepare the MNIST dataset for the following example. The complete code for this example is located at [`examples/mnist/mnist.jl`](examples/mnist/mnist.jl). See below for detailed documentation of other tutorials and user's guide.

```julia
using Mocha

data  = HDF5DataLayer(name="train-data",source="train-data-list.txt",batch_size=64)
conv  = ConvolutionLayer(name="conv1",n_filter=20,kernel=(5,5),bottoms=[:data],tops=[:conv])
pool  = PoolingLayer(name="pool1",kernel=(2,2),stride=(2,2),bottoms=[:conv],tops=[:pool])
conv2 = ConvolutionLayer(name="conv2",n_filter=50,kernel=(5,5),bottoms=[:pool],tops=[:conv2])
pool2 = PoolingLayer(name="pool2",kernel=(2,2),stride=(2,2),bottoms=[:conv2],tops=[:pool2])
fc1   = InnerProductLayer(name="ip1",output_dim=500,neuron=Neurons.ReLU(),bottoms=[:pool2],
                          tops=[:ip1])
fc2   = InnerProductLayer(name="ip2",output_dim=10,bottoms=[:ip1],tops=[:ip2])
loss  = SoftmaxLossLayer(name="loss",bottoms=[:ip2,:label])

sys = System(CuDNNBackend())
init(sys)

common_layers = [conv, pool, conv2, pool2, fc1, fc2]
net = Net("MNIST-train", sys, [data, common_layers..., loss])

params = SolverParameters(max_iter=10000, regu_coef=0.0005, momentum=0.9,
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75))
solver = SGD(params)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot("snapshots", auto_load=true),
    every_n_iter=5000)

# show performance on test data every 1000 iterations
data_test = HDF5DataLayer(name="test-data",source="test-data-list.txt",batch_size=100)
accuracy = AccuracyLayer(name="test-accuracy",bottoms=[:ip2, :label])
test_net = Net("MNIST-test", sys, [data_test, common_layers..., accuracy])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(sys)
```

## Documentation

The Mocha documentation is hosted on [readthedocs.org](http://mochajl.readthedocs.org/).

