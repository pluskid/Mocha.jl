module Mocha

include("compatibility.jl")
include("logging.jl")
include("config.jl")

if Config.use_native_extension
  include("native.jl")
end

include("macros.jl")
include("base.jl")
include("exception.jl")

include("utils/blas.jl")
include("utils/math.jl")
include("utils/io.jl")
include("utils/tensor.jl")
include("utils/ref-count.jl")

if Config.use_native_extension
  include("utils/im2col-native.jl")
else
  include("utils/im2col.jl")
end

include("backend.jl")
include("blob.jl")

if Config.use_cuda
  include("cuda/cuda.jl")
  include("cuda/cublas.jl")
  include("cuda/cudnn.jl")
  include("cuda/backend.jl")
  include("cuda/blob.jl")
  include("cuda/utils/math.jl")
  include("cuda/utils/padded-copy.jl")
  include("cuda/utils/shifted-copy.jl")
end

include("initializers.jl")
include("regularizers.jl")
include("constraints.jl")
include("neurons.jl")

if Config.use_cuda
  include("cuda/regularizers.jl")
  include("cuda/constraints.jl")
  include("cuda/neurons.jl")
end

include("pooling-functions.jl")
include("parameter.jl")

include("data-transformers.jl")
if Config.use_cuda
  include("cuda/data-transformers.jl")
end

include("layers.jl")
if Config.use_cuda
  include("cuda/layers.jl")
end

if Config.use_native_extension
  include("layers/pooling/native-impl.jl")
else
  include("layers/pooling/julia-impl.jl")
end
include("layers/pooling/channel-pooling.jl")

include("net.jl")


include("solvers.jl")
include("coffee-break.jl")
include("solvers/sgd-common.jl") # for SGD and Nesterov
include("solvers/policies.jl")
include("solvers/sgd.jl")
include("solvers/nesterov.jl")
include("solvers/adam.jl")

if Config.use_cuda
  include("cuda/solvers.jl")
end

include("utils/gradient-checking.jl")
include("utils/graphviz.jl")

end # module
