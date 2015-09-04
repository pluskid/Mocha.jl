function update_parameters!(net::Net{GPUBackend}, method::Adam,
                            alpha::Float64, epsilon::Float64, beta1::Float64, beta2::Float64,
                            m, v, gradient, param_blob, t, data_type)

  # update biased gradient moment estimates, m and v
  # m_t <- beta1*m_{t-1} + (1-beta1)*g_t

  CuBLAS.scal(net.backend.cublas_ctx, length(m), convert(data_type, beta1), m.ptr, 1)
  CuBLAS.axpy(net.backend.cublas_ctx, length(m), convert(data_type, 1-beta1), gradient.ptr, 1, m.ptr, 1)

  # now we need g2 = g.^2
  # Are we better to preallocate (heavy memory usage) or allocate each time (slower?)
  tmp = make_blob(net.backend, data_type, size(gradient))
  copy!(tmp, gradient)
  CuVec.pow!(net.backend, tmp, 2)
  CuBLAS.scal(net.backend.cublas_ctx, length(v), convert(data_type, beta2), tmp.ptr, 1)
  CuBLAS.axpy(net.backend.cublas_ctx, length(v), convert(data_type, 1-beta2), tmp.ptr, 1, v.ptr, 1)

  # Correct bias and calculate effective stepsize for timestep t (this is just scalar arithmetic)
  alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)

  # Update param <- param - alpha_t * m / (sqrt(v) + epsilon)
  copy!(tmp, v) # we'll reuse tmp for this to save reallocation
  CuVec.pow!(net.backend, tmp, convert(data_type, 0.5)) # tmp <- sqrt(v)
  CuVec.add_scal!(net.backend, tmp, convert(data_type, epsilon)) # tmp += epsilon

  CuVec.div2!(net.backend, m, tmp) # tmp = m/tmp

  CuBLAS.axpy(net.backend.cublas_ctx, length(param_blob), convert(data_type, -alpha_t), tmp.ptr, 1, param_blob.ptr, 1)

  destroy(tmp)
end
