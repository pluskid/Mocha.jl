function update_parameters!(net::Net{GPUBackend}, method::SGD, learning_rate, momentum, param_blob, hist_blob, gradient, data_type)
  # hist_blob = momentum * hist_blob
  CuBLAS.scal(net.backend.cublas_ctx, length(hist_blob), convert(data_type, momentum),
      get_ptr(hist_blob), 1)
  # hist_blob = -learning_rate * gradient + hist_blob
  CuBLAS.axpy(net.backend.cublas_ctx, length(hist_blob), convert(data_type, -learning_rate),
      get_ptr(gradient), 1, get_ptr(hist_blob), 1)

  # update parameter
  # param_blob = param_blob + hist_blob
  CuBLAS.axpy(net.backend.cublas_ctx, length(hist_blob), convert(data_type, 1),
      get_ptr(hist_blob), 1, get_ptr(param_blob), 1)
end
