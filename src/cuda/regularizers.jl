############################################################
# L2 regularization
############################################################
function forward(sys::System{CuDNNBackend}, regu :: L2Regu, param :: Blob)
  return regu.coefficient * CuBLAS.dot(sys.backend.cublas_ctx, eltype(param), length(param),
      param.ptr, 1, param.ptr, 1)
end
function backward(sys::System{CuDNNBackend}, regu :: L2Regu, param :: Blob, gradient :: Blob)
    CuBLAS.axpy(sys.backend.cublas_ctx, length(param),
        convert(eltype(param), regu.coefficient), param.ptr, 1, gradient.ptr, 1)
end

