const test_cpu   = false
const test_cudnn = true

if test_cudnn
  ENV["MOCHA_USE_CUDA"] = "true"
end

using Mocha
using Base.Test

if test_cpu
  backend_cpu = CPUBackend()
  sys_cpu     = System(backend_cpu)
  init(sys_cpu)
end

if test_cudnn
  backend_cudnn = CuDNNBackend()
  sys_cudnn     = System(backend_cudnn)
  init(sys_cudnn)
end


############################################################
# Utilities functions
############################################################
include("utils/blas.jl")

if test_cudnn
  include("cuda/cuvec.jl")
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
#-- Statistics Layers
include("layers/accuracy.jl")

#-- Data layers
include("layers/hdf5-data.jl")
include("layers/memory-data.jl")

#-- Computation Layers
include("layers/inner-product.jl")
include("layers/convolution.jl")
include("layers/pooling.jl")
include("layers/softmax.jl")
include("layers/power.jl")
include("layers/split.jl")
include("layers/element-wise.jl")
include("layers/channel-pooling.jl")
include("layers/lrn.jl")

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
