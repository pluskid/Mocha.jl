function update_parameters(net::Net{CuDNNBackend}, solver::SGD, learning_rate, state, param_blob, blob, gradient, data_type)
  # blob = net.sys.momentum * blob
  CuBLAS.scal(net.sys.backend.cublas_ctx, length(blob), convert(data_type, solver.params.momentum),
      blob.ptr, 1)
  # blob = - net.sys.learning_rate * gradient + blob
  CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, -learning_rate),
      gradient.ptr, 1, blob.ptr, 1)

  # update parameter
  # param_blob += blob
  CuBLAS.axpy(net.sys.backend.cublas_ctx, length(blob), convert(data_type, 1),
      blob.ptr, 1, param_blob.ptr, 1)
end

