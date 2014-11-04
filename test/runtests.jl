using Mocha
using Base.Test

global test_cpu   = true
global test_cudnn = false

if test_cpu
  backend_cpu = CPUBackend()
  sys_cpu     = System(backend_cpu, 0.0005, 0.01, 0.9, 5000)
  init(sys_cpu)
end

if test_cudnn
  backend_cudnn = CuDNNBackend()
  sys_cudnn     = System(backend_cudnn, 0.0005, 0.01, 0.9, 5000)
  init(sys_cudnn)
end

if test_cudnn
  include("cuda/cublas.jl")
end

############################################################
# Layers
############################################################
#-- Data layers
include("layers/hdf5-data.jl")
include("layers/memory-data.jl")

#-- Computation Layers
include("layers/inner-product.jl")
#include("layers/convolution.jl")

#-- Loss Layers
include("layers/square-loss.jl")

if test_cudnn
  shutdown(sys_cudnn)
end
if test_cpu
  shutdown(sys_cpu)
end
