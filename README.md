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

## Documentation

The Mocha documentation is hosted on [readthedocs.org](http://mochajl.readthedocs.org/).

