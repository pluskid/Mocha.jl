# Mocha

[![Build Status](https://travis-ci.org/pluskid/Mocha.jl.svg?branch=master)](https://travis-ci.org/pluskid/Mocha.jl)

Deep Learning framework for julia. Multiple backends are supported, currently including CUDA+CuDNN, and CPU (working in progress). See [`examples/mnist/mnist.jl`](examples/mnist/mnist.jl) for an example of LeNet training on MNIST.

CPU backend and CUDA backend are supported. To use CUDA backend, one needs to

- install CUDA
- install [CuDNN](https://developer.nvidia.com/cuDNN) (currently NOT available on Mac OS X)
- compile CUDA files in `src/cuda/kernels` by `nvcc -ptx kernels.cu`
