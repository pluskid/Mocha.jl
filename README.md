# Mocha

[![Build Status](https://travis-ci.org/pluskid/Mocha.jl.svg?branch=master)](https://travis-ci.org/pluskid/Mocha.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/342vcj5lj2jyegsp?svg=true)](https://ci.appveyor.com/project/pluskid/mocha-jl)
[![Documentation Status](https://readthedocs.org/projects/mochajl/badge/?version=latest)](http://mochajl.readthedocs.org/)
[![Mocha](http://pkg.julialang.org/badges/Mocha_release.svg)](http://pkg.julialang.org/?pkg=Mocha&ver=release)


Mocha is a Deep Learning framework for [Julia](http://julialang.org/), inspired by the C++ Deep Learning framework [Caffe](http://caffe.berkeleyvision.org/). Mocha support multiple backends:

- Pure Julia CPU Backend: Implemented in pure Julia; Runs out of the box without any external dependency; Reasonably fast on small models thanks to Julia's LLVM-based just-in-time (JIT) compiler and [Performance Annotations](http://julia.readthedocs.org/en/latest/manual/performance-tips/#performance-annotations) that eliminate unnecessary bound checkings.
- CPU Backend with Native Extension: Some bottleneck computations (Convolution and Pooling) have C++ implementations. When compiled and enabled, could be slightly faster than the pure Julia backend (on the MNIST example, roughly 2 times faster, similar to the speed of Caffe's CPU backend).
- CUDA + cuDNN: An interface to NVidia® [cuDNN](https://developer.nvidia.com/cuDNN) GPU accelerated deep learning library. When run with CUDA GPU devices, could be much faster depending on the size of the problem (MNIST is roughly 20 times faster than the pure Julia backend).

## Installation

To install the release version, simply run

```
Pkg.add("Mocha")
```

in Julia console. To install the latest development version, run the following command instead:

```
Pkg.clone("https://github.com/pluskid/Mocha.jl.git")
```

Then you can run the built-in unit tests with

```
Pkg.test("Mocha")
```

to verify that everything is functioning properly on your machine.

## Hello World

Please refer to [the MNIST tutorial](http://mochajl.readthedocs.org/en/latest/tutorial/mnist.html) for how prepare the MNIST dataset. The complete code for the MNIST example is at [`examples/mnist/mnist.jl`](examples/mnist/mnist.jl). See below for detailed documentation of other tutorials and user's guide.

```julia
using Mocha

data_layer  = HDF5DataLayer(name="train-data", source=source_fns[1], batch_size=64)
conv_layer  = ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5), bottoms=[:data], tops=[:conv])
pool_layer  = PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2), bottoms=[:conv], tops=[:pool])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=50, kernel=(5,5), bottoms=[:pool], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", kernel=(2,2), stride=(2,2), bottoms=[:conv2], tops=[:pool2])
fc1_layer   = InnerProductLayer(name="ip1", output_dim=500, neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
fc2_layer   = InnerProductLayer(name="ip2", output_dim=10, bottoms=[:ip1], tops=[:ip2])
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])

sys = System(CuDNNBackend())
init(sys)

common_layers = [conv_layer, pool_layer, conv2_layer, pool2_layer, fc1_layer, fc2_layer]
net = Net("MNIST-train", sys, [data_layer, common_layers..., loss_layer])

params = SolverParameters(max_iter=10000, regu_coef=0.0005, momentum=0.9,
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75))
solver = SGD(params)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver,
    Snapshot("snapshots", auto_load=true),
    every_n_iter=5000)

# show performance on test data every 1000 iterations
data_layer_test = HDF5DataLayer(name="test-data", source=source_fns[2], batch_size=100)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:ip2, :label])
test_net = Net("MNIST-test", sys, [data_layer_test, common_layers..., acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(sys)
```

## Documentation

The Mocha documentation is hosted on [readthedocs.org](http://mochajl.readthedocs.org/).

