function forward(sys::System{CuDNNBackend}, state::SubMeanState, input::Blob)
  width, height, channels, num = size(input)
  fea_dim = width*height*channels
  CuBLAS.gemm(sys.backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, fea_dim, num, 1, convert(eltype(input), -1),
      state.mean_blob.ptr, fea_dim, state.multiplier.ptr, 1, convert(eltype(input), 1),
      input.ptr, fea_dim)
end

function forward(sys::System{CuDNNBackend}, state::ScaleState, input::Blob)
  CuVec.mul_scal!(sys, input, state.scale)
end
