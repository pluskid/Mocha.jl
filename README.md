# Mocha

[![Build Status](https://img.shields.io/travis/pluskid/Mocha.jl.svg?style=flat&branch=master)](https://travis-ci.org/pluskid/Mocha.jl)
[![Documentation Status](https://readthedocs.org/projects/mochajl/badge/?version=latest)](http://mochajl.readthedocs.org/)
[![Mocha](http://pkg.julialang.org/badges/Mocha_release.svg)](http://pkg.julialang.org/?pkg=Mocha&ver=release)
[![Coverage Status](https://img.shields.io/coveralls/pluskid/Mocha.jl.svg?style=flat)](https://coveralls.io/r/pluskid/Mocha.jl?branch=master)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
<!--[![Build status](https://ci.appveyor.com/api/projects/status/342vcj5lj2jyegsp?svg=true)](https://ci.appveyor.com/project/pluskid/mocha-jl)-->

[Tutorials](http://mochajl.readthedocs.org/en/latest/#tutorials) | [Documentation](http://mochajl.readthedocs.org/) | [Release Notes](NEWS.md) | [Roadmap](https://github.com/pluskid/Mocha.jl/issues/22) | [Issues](https://github.com/pluskid/Mocha.jl/issues)

Mocha is a Deep Learning framework for [Julia](http://julialang.org/), inspired by the C++ framework [Caffe](http://caffe.berkeleyvision.org/). Efficient implementations of general stochastic gradient solvers and common layers in Mocha could be used to train deep / shallow (convolutional) neural networks, with (optional) unsupervised pre-training via (stacked) auto-encoders. Some highlights:

- **Modular Architecture**: Mocha has a clean architecture with isolated components like network layers, activation functions, solvers, regularizers, initializers, etc. Built-in components are sufficient for typical deep (convolutional) neural network applications and more are being added in each release. All of them could be easily extended by adding custom sub-types.
- **High-level Interface**: Mocha is written in [Julia](http://julialang.org/), a high-level dynamic programming language designed for scientific computing. Combining with the expressive power of Julia and other its package eco-system, playing with deep neural networks in Mocha is easy and intuitive. See for example our IJulia Notebook example of [using a pre-trained imagenet model to do image classification](http://nbviewer.ipython.org/github/pluskid/Mocha.jl/blob/master/examples/ijulia/ilsvrc12/imagenet-classifier.ipynb).
- **Portability and Speed**: Mocha comes with multiple backend that could be switched transparently.
  - The *pure Julia backend* is portable -- it runs on any platform that support Julia. This is reasonably fast on small models thanks to Julia's LLVM-based just-in-time (JIT) compiler and [Performance Annotations](http://julia.readthedocs.org/en/latest/manual/performance-tips/#performance-annotations), and could be very useful for prototyping.
  - The *native extension backend* could be turned on when a C++ compiler is available. It runs 2~3 times faster than the pure Julia backend.
  - The *GPU backend* uses NVidia® [cuDNN](https://developer.nvidia.com/cuDNN), cuBLAS and customized CUDA kernels to provide highly efficient computation. 20~30 times or even more speedup could be observed on a modern GPU device, especially on larger models.
- **Compatibility**: Mocha uses the widely adopted HDF5 format to store both datasets and model snapshots, making it easy to inter-operate with Matlab, Python (numpy) and other existing computational tools. Mocha also provides tools to import trained model snapshots from Caffe.
- **Correctness**: the computational components in Mocha in all backends are extensively covered by unit-tests.
- **Open Source**: Mocha is licensed under [the MIT "Expat" License](LICENSE.md).

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

backend = GPUBackend()
init(backend)

common_layers = [conv, pool, conv2, pool2, fc1, fc2]
net = Net("MNIST-train", backend, [data, common_layers..., loss])

exp_dir = "snapshots"
params = SolverParameters(max_iter=10000, regu_coef=0.0005,
    mom_policy=MomPolicy.Fixed(0.9),
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
    load_from=exp_dir)
solver = SGD(params)

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
data_test = HDF5DataLayer(name="test-data",source="test-data-list.txt",batch_size=100)
accuracy = AccuracyLayer(name="test-accuracy",bottoms=[:ip2, :label])
test_net = Net("MNIST-test", backend, [data_test, common_layers..., accuracy])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)
```

## Documentation

The Mocha documentation is hosted at [readthedocs.org](http://mochajl.readthedocs.org/).

