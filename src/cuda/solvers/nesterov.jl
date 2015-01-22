function update_parameters(net::Net{GPUBackend}, solver::Nesterov, learning_rate,
    last_momentum, momentum, param_blob, hist_blob, gradient, data_type)

  # param_blob += -last_momentum* hist_blob (update with vt-1)
  CuBLAS.axpy(net.backend.cublas_ctx, length(hist_blob), convert(data_type, -last_momentum), hist_blob.ptr, 1, param_blob.ptr, 1)

  # hist_blob = last_momentum * hist_blob
  CuBLAS.scal(net.backend.cublas_ctx, length(hist_blob), convert(data_type, last_momentum), hist_blob.ptr, 1)

  # hist_blob = -learning_rate * gradient + hist_blob (calc vt)
  CuBLAS.axpy(net.backend.cublas_ctx, length(hist_blob), convert(data_type, -learning_rate), gradient.ptr, 1, hist_blob.ptr, 1)


  # param_blob += (1+momentum) * hist_blob (update with vt)
  CuBLAS.axpy(net.backend.cublas_ctx, length(hist_blob), convert(data_type, 1 + momentum), hist_blob.ptr, 1, param_blob.ptr, 1)
end
