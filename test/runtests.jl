using Mocha
using Base.Test

backend_cudnn = CuDNNBackend()
sys_cudnn = System(backend_cudnn, 0.0005, 0.01, 0.9, 5000)
init(sys_cudnn)

include("layers/hdf5-data.jl")
include("layers/memory-data.jl")

include("layers/inner-product.jl")

shutdown(sys_cudnn)
