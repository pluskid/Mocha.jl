**Update Dec. 2018**: Mocha.jl is now deprecated. The latest version works with Julia v0.6. If you have existing legacy codebase with Mocha that you want to updates for Julia v1.0, the pull request [255](https://github.com/pluskid/Mocha.jl/pull/255) contains fixes for CPU backend only that have all the unit tests passed under Julia v1.0. 

The development of Mocha.jl happens in relative early days of Julia. Now that both Julia and the ecosystem has evolved significantly, and with some exciting new tech such as writing GPU kernels directly in Julia and general auto-differentiation supports, the Mocha codebase becomes excessively old and primitive. Reworking Mocha with new technologies requires some non-trivial efforts, and new exciting solutions already exist nowadays, it is a good time for the retirement of Mocha.jl.

If you are interested in doing deep learning with Julia, please check out some alternative packages that are more up-to-date and actively maintained. In particular, there are [Knet.jl](https://github.com/denizyuret/Knet.jl) and [Flux.jl](https://github.com/FluxML/Flux.jl) for pure-Julia solutions, and [MXNet.jl](https://github.com/dmlc/MXNet.jl) and [Tensorflow.jl](https://github.com/malmaud/TensorFlow.jl) for wrapper to existing deep learning systems.

# Mocha

[![Build Status](https://img.shields.io/travis/pluskid/Mocha.jl.svg?style=flat&branch=master)](https://travis-ci.org/pluskid/Mocha.jl)
[![Documentation Status](https://readthedocs.org/projects/mochajl/badge/?version=latest)](http://mochajl.readthedocs.org/)
[![Mocha](http://pkg.julialang.org/badges/Mocha_0.6.svg)](http://pkg.julialang.org/?pkg=Mocha&ver=0.6)
[![Coverage Status](https://img.shields.io/coveralls/pluskid/Mocha.jl.svg?style=flat)](https://coveralls.io/r/pluskid/Mocha.jl?branch=master)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
<!--[![Build status](https://ci.appveyor.com/api/projects/status/342vcj5lj2jyegsp?svg=true)](https://ci.appveyor.com/project/pluskid/mocha-jl)-->

[Tutorials](http://mochajl.readthedocs.org/en/latest/#tutorials) | [Documentation](http://mochajl.readthedocs.org/) | [Release Notes](NEWS.md) | [Roadmap](https://github.com/pluskid/Mocha.jl/issues/22) | [Issues](https://github.com/pluskid/Mocha.jl/issues)

Mocha is a Deep Learning framework for [Julia](http://julialang.org/), inspired by the C++ framework [Caffe](http://caffe.berkeleyvision.org/). Efficient implementations of general stochastic gradient solvers and common layers in Mocha can be used to train deep / shallow (convolutional) neural networks, with (optional) unsupervised pre-training via (stacked) auto-encoders. Some highlights:

- **Modular Architecture**: Mocha has a clean architecture with isolated components like network layers, activation functions, solvers, regularizers, initializers, etc. Built-in components are sufficient for typical deep (convolutional) neural network applications and more are being added in each release. All of them can be easily extended by adding custom sub-types.
- **High-level Interface**: Mocha is written in [Julia](http://julialang.org/), a high-level dynamic programming language designed for scientific computing. Combining with the expressive power of Julia and its package eco-system, playing with deep neural networks in Mocha is easy and intuitive. See for example our IJulia Notebook example of [using a pre-trained imagenet model to do image classification](http://nbviewer.ipython.org/github/pluskid/Mocha.jl/blob/master/examples/ijulia/ilsvrc12/imagenet-classifier.ipynb).
- **Portability and Speed**: Mocha comes with multiple backends that can be switched transparently.
  - The *pure Julia backend* is portable -- it runs on any platform that supports Julia. This is reasonably fast on small models thanks to Julia's LLVM-based just-in-time (JIT) compiler and [Performance Annotations](http://julia.readthedocs.org/en/latest/manual/performance-tips/#performance-annotations), and can be very useful for prototyping.
  - The *native extension backend* can be turned on when a C++ compiler is available. It runs 2~3 times faster than the pure Julia backend.
  - The *GPU backend* uses NVidia® [cuDNN](https://developer.nvidia.com/cuDNN), cuBLAS and customized CUDA kernels to provide highly efficient computation. 20~30 times or even more speedup could be observed on a modern GPU device, especially on larger models.
- **Compatibility**: Mocha uses the widely adopted HDF5 format to store both datasets and model snapshots, making it easy to inter-operate with Matlab, Python (numpy) and other existing computational tools. Mocha also provides tools to import trained model snapshots from Caffe.
- **Correctness**: the computational components in Mocha in all backends are extensively covered by unit-tests.
- **Open Source**: Mocha is licensed under [the MIT "Expat" License](LICENSE.md).

## Installation

To install the release version, simply run

```julia
Pkg.add("Mocha")
```

on the Julia console. To install the latest development version, run the following command instead:

```julia
Pkg.clone("https://github.com/pluskid/Mocha.jl.git")
```

Then you can run the built-in unit tests with

```julia
Pkg.test("Mocha")
```

to verify that everything is functioning properly on your machine.

## Hello World

Please refer to [the MNIST tutorial](http://mochajl.readthedocs.org/en/latest/tutorial/mnist.html) on how to prepare the MNIST dataset for the following example. The complete code for this example is located at [`examples/mnist/mnist.jl`](examples/mnist/mnist.jl). See below for detailed documentation of other tutorials and user guide.

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

backend = DefaultBackend()
init(backend)

common_layers = [conv, pool, conv2, pool2, fc1, fc2]
net = Net("MNIST-train", backend, [data, common_layers..., loss])

exp_dir = "snapshots"
solver_method = SGD()
params = make_solver_parameters(solver_method, max_iter=10000, regu_coef=0.0005,
    mom_policy=MomPolicy.Fixed(0.9),
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
    load_from=exp_dir)
solver = Solver(solver_method, params)

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
