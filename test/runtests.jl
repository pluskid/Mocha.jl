using Mocha
using Base.Test

global test_cpu   = false
global test_cudnn = true

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
  include("cuda/mocha.jl")
  include("cuda/cublas.jl")
end

############################################################
# Activation Functions
############################################################
include("neurons/relu.jl")

############################################################
# Regularizers
############################################################
include("regularizers/l2.jl")

############################################################
# Layers
############################################################
#-- Data layers
include("layers/hdf5-data.jl")
include("layers/memory-data.jl")

#-- Computation Layers
include("layers/inner-product.jl")
include("layers/convolution.jl")
include("layers/pooling.jl")
include("layers/softmax.jl")

#-- Loss Layers
include("layers/square-loss.jl")
include("layers/multinomial-logistic-loss.jl")
include("layers/softmax-loss.jl")

if test_cudnn
  shutdown(sys_cudnn)
end
if test_cpu
  shutdown(sys_cpu)
end
