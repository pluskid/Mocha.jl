function setup_etc(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
  state.etc = make_blob(backend, eltype(inputs[1]), size(inputs[1]))
end
function shutdown_etc(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState)
  destroy(state.etc)
end
function forward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]

  forward(backend, state.softmax, Blob[pred])
  copy!(state.etc, state.softmax.blobs[1])
  prob = state.etc

  dims = size(prob)
  data_type = eltype(prob)

  CuVec.log!(backend, prob)
  loss = -CuBLAS.dot(backend.cublas_ctx, data_type, length(prob), prob.ptr, 1, label.ptr, 1)
  state.loss = state.layer.weight * loss / (prod(dims) / dims[state.op_dim])
end

function backward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]

  if isa(diff, CuTensorBlob)
    label = inputs[2]
    copy!(diff, state.softmax.blobs[1])
    data_type = eltype(diff)
    dims = size(diff)

    CuBLAS.axpy(backend.cublas_ctx, length(diff), -one(data_type), label.ptr, 1, diff.ptr, 1)
    CuBLAS.scal(backend.cublas_ctx, length(diff), convert(data_type, state.layer.weight * dims[state.op_dim]/prod(dims)), diff.ptr, 1)
  end
end

