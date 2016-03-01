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
  loss = -CuBLAS.dot(get_cublas_ctx(backend), data_type, length(prob), get_ptr(prob), 1, get_ptr(label), 1)
  state.loss = state.layer.weight * loss / (prod(dims) / dims[state.op_dim])
end

function backward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]

  if isa(diff, CuTensorBlob)
    label = inputs[2]
    copy!(diff, state.softmax.blobs[1])
    data_type = eltype(diff)
    dims = size(diff)

    CuBLAS.axpy(get_cublas_ctx(backend), length(diff), -one(data_type), get_ptr(label), 1, get_ptr(diff), 1)
    CuBLAS.scal(get_cublas_ctx(backend), length(diff), convert(data_type, state.layer.weight * dims[state.op_dim]/prod(dims)), get_ptr(diff), 1)
  end
end

