function forward(backend::GPUBackend, state::SquareLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)
  n = length(pred)

  pred_arr = to_array(pred)
  if any(isnan(pred_arr))
    error("NaN in pred")
  end
  label_arr = to_array(label)
  if any(isnan(label_arr))
    error("NaN in label")
  end

  copy!(state.pred_copy, pred)
  CuBLAS.axpy(backend.cublas_ctx, n, convert(data_type, -1), label.ptr, 1, state.pred_copy.ptr, 1)
  state.loss = 0.5/get_num(pred)*CuBLAS.dot(backend.cublas_ctx, data_type, n, state.pred_copy.ptr, 1, state.pred_copy.ptr, 1)
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
    CuBLAS.axpy(backend.cublas_ctx, n, convert(data_type, 1.0/num), pred.ptr, 1, diff.ptr, 1)
    CuBLAS.axpy(backend.cublas_ctx, n, convert(data_type, -1.0/num), label.ptr, 1, diff.ptr, 1)
  end
end
