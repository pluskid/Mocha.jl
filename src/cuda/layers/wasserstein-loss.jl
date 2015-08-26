function sinkhorn(backend::GPUBackend, state::WassersteinLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)

  pred_size = get_fea_size(pred)
  pred_num  = get_num(pred)
  label_size= get_fea_size(label)

  # init as uniform distribution
  copy!(state.u, ones(data_type, pred_size, pred_num) / pred_size);
  u = state.u
  a = pred
  b = label
  K = state.K

  if isempty(state.tmps)
    state.tmps = Blob[
      make_blob(backend, data_type, size(b)),
      make_blob(backend, data_type, size(a)),
      make_blob(backend, ones(data_type, size(a))/pred_num)
    ]
  end

  for iter = 1:state.layer.sinkhorn_iter
    # u = a ./ (K * (b./(u'*K)'))

    # tmps[1] = K' * u
    CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_T, CuBLAS.OP_N,
        label_size, pred_num, pred_size, one(data_type), K.ptr, pred_size,
        u.ptr, pred_size, zero(data_type), state.tmps[1].ptr, label_size)

    # tmps[1] = b ./ tmps[1]
    CuVec.div2!(backend, b, state.tmps[1])

    # tmps[2] = K * tmps[1]
    CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N,
        pred_size, pred_num, label_size, one(data_type), K.ptr, pred_size,
        state.tmps[1].ptr, label_size, zero(data_type), state.tmps[2].ptr, pred_size)

    # tmps[2] = a ./ tmps[2]
    CuVec.div2!(backend, a, state.tmps[2])

    # u = tmps[2]
    copy!(u, state.tmps[2])
  end

  # compute objective function
  #-------------------------------------
  # v = b ./ (K'*u)

  # tmps[1] = K' * u
  CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_T, CuBLAS.OP_N,
      label_size, pred_num, pred_size, one(data_type), K.ptr, pred_size,
      u.ptr, pred_size, zero(data_type), state.tmps[1].ptr, label_size)

  # tmps[1] = b ./ tmps[1]
  CuVec.div2!(backend, b, state.tmps[1])

  #-------------------------------------
  # loss = sum(u .* (KM * v)) / pred_num

  # tmps[2] = KM * tmp[1]
  CuBLAS.gemm(backend.cublas_ctx, CuBLAS.OP_N, CuBLAS.OP_N,
      pred_size, pred_num, label_size, one(data_type), state.KM.ptr, pred_size,
      state.tmps[1].ptr, label_size, zero(data_type), state.tmps[2].ptr, pred_size)

  # tmps[2] = u .* tmps[2]
  CuVec.mul!(backend, state.tmps[2], u)

  # tmps[3] == ones/pred_num
  # loss = sum(tmps[2]) / pred_num
  state.loss = CuBLAS.dot(backend.cublas_ctx, data_type, length(state.tmps[2]),
      state.tmps[2].ptr, 1, state.tmps[3].ptr, 1)

  # compute gradient
  copy!(state.alpha, u)
  CuVec.log!(backend, state.alpha)
  CuBLAS.scal(backend.cublas_ctx, length(state.alpha), convert(data_type, 1.0/state.layer.lambda/pred_num),
      state.alpha.ptr, 1)
end

