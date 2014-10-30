type CuDNNBackend <: Backend
  initialized:: Bool
  cu_ctx     :: CuContext
  cublas_ctx :: CuBLAS.Handle
  cudnn_ctx  :: CuDNN.Handle

  CuDNNBackend() = new(false) # everything will be initialized later
end

function init(backend::CuDNNBackend)
  @assert backend.initialized == false

  dev = CuDevice(0)
  backend.cu_ctx = create_context(dev)
  backend.cublas_ctx = CuBLAS.create()
  backend.cudnn_ctx = CuDNN.create()
end

function shutdown(backend::CuDNNBackend)
  @assert backend.initialized == true

  destroy(backend.cu_ctx)
  CuBLAS.destroy(backend.cublas_ctx)
  CuDNN.destroy(backend.cudnn_ctx)
  abckend.initialized = false
end


