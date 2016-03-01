function forward(backend::GPUBackend, state::SubMeanState, input::Blob)
  fea_dim = get_fea_size(input)
  num     = get_num(input)
  CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_N, CuBLAS.OP_N, fea_dim, num, 1, convert(eltype(input), -1),
      get_ptr(state.mean_blob), fea_dim, get_ptr(state.multiplier), 1, convert(eltype(input), 1),
      get_ptr(input), fea_dim)
end

function forward(backend::GPUBackend, state::ScaleState, input::Blob)
  CuVec.mul_scal!(backend, input, state.scale)
end
