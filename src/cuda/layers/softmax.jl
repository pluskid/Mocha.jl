type CuDNNSoftmaxState
  inputs_desc  :: Vector{CuDNN.Tensor4dDescriptor}
  outputs_desc :: Vector{CuDNN.Tensor4dDescriptor}
end

function setup_etc(backend::GPUBackend, layer::SoftmaxLayer, dims::Vector{Int}, data_type, inputs)
  inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
  outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
  for i = 1:length(inputs)
    dim_sp, dim_prob, dim_num = split_dims(inputs[i], dims[i])
    inputs_desc[i] = CuDNN.create_tensor4d_descriptor(data_type, (1,dim_sp,dim_prob,dim_num))
    outputs_desc[i] = CuDNN.create_tensor4d_descriptor(data_type, (1,dim_sp,dim_prob,dim_num))
  end
  etc = CuDNNSoftmaxState(inputs_desc, outputs_desc)
  return etc
end
function shutdown(backend::GPUBackend, state::SoftmaxLayerState)
  map(destroy, state.blobs)
  map(CuDNN.destroy_tensor4d_descriptor, state.etc.inputs_desc)
  map(CuDNN.destroy_tensor4d_descriptor, state.etc.outputs_desc)
end

function forward(backend::GPUBackend, state::SoftmaxLayerState, inputs::Vector{Blob})
  alpha = one(eltype(inputs[1]))
  beta = zero(eltype(inputs[1]))
  for i = 1:length(inputs)
    CuDNN.softmax_forward(backend.cudnn_ctx, CuDNN.CUDNN_SOFTMAX_ACCURATE,
        CuDNN.CUDNN_SOFTMAX_MODE_CHANNEL, alpha, state.etc.inputs_desc[i], inputs[i].ptr,
        beta, state.etc.outputs_desc[i], state.blobs[i].ptr)
  end
end

function backward(backend::GPUBackend, state::SoftmaxLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  alpha = one(eltype(inputs[1]))
  beta = zero(eltype(inputs[1]))

  for i = 1:length(inputs)
    diff = diffs[i]
    if !isa(diff, NullBlob)
      CuDNN.softmax_backward(backend.cudnn_ctx, CuDNN.CUDNN_SOFTMAX_ACCURATE,
          CuDNN.CUDNN_SOFTMAX_MODE_CHANNEL, alpha, state.etc.outputs_desc[i], state.blobs[i].ptr,
          state.etc.outputs_desc[i], state.blobs_diff[i].ptr, beta, state.etc.inputs_desc[i], diff.ptr)
    end
  end
end
