# Mocha

[![Build Status](https://travis-ci.org/pluskid/Mocha.jl.svg?branch=master)](https://travis-ci.org/pluskid/Mocha.jl)

Mocha is a Deep Learning framework for [Julia](http://julialang.org/), inspired by the C++ Deep Learning framework [Caffe](http://caffe.berkeleyvision.org/). Mocha support multiple backends:

- Pure Julia CPU Backend: Implemented in pure Julia; Runs out of the box without any external dependency; Already pretty fast thanks to Julia's LLVM-based just-in-time (JIT) compiler and [Performance Annotations](http://julia.readthedocs.org/en/latest/manual/performance-tips/#performance-annotations) that eliminate unnecessary bound checkings.
- CPU Backend with Native Extension: Some bottleneck computations (Convolution and Pooling) have C++ implementations. When compiled and enabled, could be slightly faster than the pure Julia backend (on the MNIST example, roughly 2 times faster, similar to the speed of Caffe's CPU backend).
- CUDA + cuDNN: An interface to NVidia® [cuDNN](https://developer.nvidia.com/cuDNN) GPU accelerated deep learning library. When run with CUDA GPU devices, could be much faster depending on the size of the problem (MNIST is roughly 20 times faster than the pure Julia backend).

## Installation

To install the release version (nothing released yet), simply run

```
Pkg.add("Mocha")
```

in Julia console. To install the latest development version (currently the only way of installing), run the following command instead:

```
Pkg.clone("https://github.com/pluskid/Mocha.jl.git")
```

Dependencies should be automatically installed. However, currently, Mocha depends on a feature of [HDF5.jl](https://github.com/timholy/HDF5.jl) that is only available in the development version. To checkout the development version of HDF5.jl, run

```
Pkg.checkout("HDF5")
```

You can run the built-in unit tests with

```
Pkg.test("Mocha")
```

to verify that everything is functioning properly on your machine.
