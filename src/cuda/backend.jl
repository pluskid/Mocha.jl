export CuDNNBackend

type MochaKernels
  mod :: CUDA.CuModule

  # implemented kernels
  logistic_loss_forward_float  :: CUDA.CuFunction
  logistic_loss_forward_double :: CUDA.CuFunction
  softmax_loss_backward_float  :: CUDA.CuFunction
  softmax_loss_backward_double :: CUDA.CuFunction
  relu_forward_float           :: CUDA.CuFunction
  relu_forward_double          :: CUDA.CuFunction
  relu_backward_float          :: CUDA.CuFunction
  relu_backward_double         :: CUDA.CuFunction
  accuracy_forward_float       :: CUDA.CuFunction
  accuracy_forward_double      :: CUDA.CuFunction

  MochaKernels() = begin
    mod_path = joinpath(dirname(@__FILE__), "kernels", "kernels.ptx")
    mod = CUDA.CuModule(mod_path)
    kernels = new(mod)
  
    kernels.logistic_loss_forward_float = CUDA.CuFunction(mod, "logistic_loss_forward_float")
    kernels.logistic_loss_forward_double = CUDA.CuFunction(mod, "logistic_loss_forward_double")
    kernels.softmax_loss_backward_float = CUDA.CuFunction(mod, "softmax_loss_backward_float")
    kernels.softmax_loss_backward_double = CUDA.CuFunction(mod, "softmax_loss_backward_double")
    kernels.relu_forward_float = CUDA.CuFunction(mod, "relu_forward_float")
    kernels.relu_forward_double = CUDA.CuFunction(mod, "relu_forward_double")
    kernels.relu_backward_float = CUDA.CuFunction(mod, "relu_backward_float")
    kernels.relu_backward_double = CUDA.CuFunction(mod, "relu_backward_double")
    kernels.accuracy_forward_float = CUDA.CuFunction(mod, "accuracy_forward_float")
    kernels.accuracy_forward_double = CUDA.CuFunction(mod, "accuracy_forward_double")

    return kernels
  end
end
function shutdown(mocha :: MochaKernels)
  CUDA.unload(mocha.mod)
end

type CuDNNBackend <: Backend
  initialized:: Bool
  cu_ctx     :: CUDA.CuContext
  cublas_ctx :: CuBLAS.Handle
  cudnn_ctx  :: CuDNN.Handle

  mocha      :: MochaKernels

  CuDNNBackend() = new(false) # everything will be initialized later
end

function init(backend::CuDNNBackend)
  @assert backend.initialized == false

  @info("Initializing CuDNN backend...")
  CUDA.init()
  dev = CUDA.CuDevice(0)
  backend.cu_ctx = CUDA.create_context(dev)
  backend.cublas_ctx = CuBLAS.create()
  backend.cudnn_ctx = CuDNN.create()
  backend.mocha = MochaKernels()
  backend.initialized = true
  info("CuDNN backend initialized!")
end

function shutdown(backend::CuDNNBackend)
  @assert backend.initialized == true

  @info("Shutting down CuDNN backend...")
  # NOTE: destroy should be in reverse order of init
  shutdown(backend.mocha)
  CuDNN.destroy(backend.cudnn_ctx)
  CuBLAS.destroy(backend.cublas_ctx)
  CUDA.destroy(backend.cu_ctx)
  backend.initialized = false
  @info("CuDNN Backend shutdown finished!")
end

