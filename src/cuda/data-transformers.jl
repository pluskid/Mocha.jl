function forward(backend::GPUBackend, state::SubMeanState, input::Blob)
  fea_dim = get_fea_size(input)
  num     = get_num(input)
  CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N, fea_dim, num, 1, convert(eltype(input), -1),
      state.mean_blob.ptr, fea_dim, state.multiplier.ptr, 1, convert(eltype(input), 1),
      input.ptr, fea_dim)
end

function forward(backend::GPUBackend, state::ScaleState, input::Blob)
  CuVec.mul_scal!(backend, input, state.scale)
end
