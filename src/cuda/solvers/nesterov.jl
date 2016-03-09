#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function update_parameters!(net::Net{GPUBackend}, method::Nesterov, learning_rate,
    last_momentum, momentum, param_blob, hist_blob, gradient, data_type)

  # param_blob += -last_momentum* hist_blob (update with vt-1)
  CuBLAS.axpy(get_cublas_ctx(net.backend), length(hist_blob), convert(data_type, -last_momentum), get_ptr(hist_blob), 1, get_ptr(param_blob), 1)

  # hist_blob = last_momentum * hist_blob
  CuBLAS.scal(get_cublas_ctx(net.backend), length(hist_blob), convert(data_type, last_momentum), get_ptr(hist_blob), 1)

  # hist_blob = -learning_rate * gradient + hist_blob (calc vt)
  CuBLAS.axpy(get_cublas_ctx(net.backend), length(hist_blob), convert(data_type, -learning_rate), get_ptr(gradient), 1, get_ptr(hist_blob), 1)


  # param_blob += (1+momentum) * hist_blob (update with vt)
  CuBLAS.axpy(get_cublas_ctx(net.backend), length(hist_blob), convert(data_type, 1 + momentum), get_ptr(hist_blob), 1, get_ptr(param_blob), 1)
end
