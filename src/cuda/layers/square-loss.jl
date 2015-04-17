function forward(backend::GPUBackend, state::SquareLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  copy!(state.pred_copy, pred)
  CuBLAS.axpy(backend.cublas_ctx, n, convert(data_type, -1), label.ptr, 1, state.pred_copy.ptr, 1)
  state.loss = state.layer.weight * 0.5/get_num(pred)*CuBLAS.dot(backend.cublas_ctx, data_type, n, state.pred_copy.ptr, 1, state.pred_copy.ptr, 1)

  # accumulate statistics
  state.loss_accum = (state.loss_accum*state.n_accum + state.loss*get_num(pred)) / (state.n_accum+get_num(pred))
  state.n_accum += get_num(pred)
end

function backward(backend::GPUBackend, state::SquareLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]
  if isa(diff, CuTensorBlob)
    pred = inputs[1]
    label = inputs[2]

    data_type = eltype(pred)
    n = length(pred)
    num = get_num(pred)

    erase!(diff)
    CuBLAS.axpy(backend.cublas_ctx, n, convert(data_type, state.layer.weight/num), pred.ptr, 1, diff.ptr, 1)
    CuBLAS.axpy(backend.cublas_ctx, n, convert(data_type, -state.layer.weight/num), label.ptr, 1, diff.ptr, 1)
  end

  # the "label" also needs gradient
  if isa(diffs[2], CuTensorBlob)
    copy!(diffs[2], diff)
    CuBLAS.scal(backend.cublas_ctx, n, -one(data_type), diffs[2].ptr, 1)
  end
end
