type CuDNNSoftmaxState
  inputs_desc  :: Vector{CuDNN.Tensor4dDescriptor}
  outputs_desc :: Vector{CuDNN.Tensor4dDescriptor}
end

function setup_etc(backend::GPUBackend, layer::SoftmaxLayer, dims::Vector{Int}, data_type, inputs)
  softmax_states = MultiGPUType(backend, CuDNNSoftmaxState)
  orig_dev = backend.cur_dev.ordinal

  for dev=1:backend.dev_count
      set_dev_id(backend, dev - 1)

      inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      for i = 1:length(inputs)
        dim_sp, dim_prob, dim_num = split_dims(inputs[i], dims[i])
        inputs_desc[i] = CuDNN.create_tensor4d_descriptor(data_type, (1,dim_sp,dim_prob,dim_num))
        outputs_desc[i] = CuDNN.create_tensor4d_descriptor(data_type, (1,dim_sp,dim_prob,dim_num))
      end
      softmax_states.elems[dev] = CuDNNSoftmaxState(inputs_desc, outputs_desc)
  end
  set_dev_id(backend, orig_dev)
  return softmax_states
end
function shutdown(backend::GPUBackend, state::SoftmaxLayerState)
  map(destroy, state.blobs)

  softmax_states = state.etc
  for etc in softmax_states.elems
    map(CuDNN.destroy_tensor4d_descriptor, etc.inputs_desc)
    map(CuDNN.destroy_tensor4d_descriptor, etc.outputs_desc)
  end
end

function forward(backend::GPUBackend, state::SoftmaxLayerState, inputs::Vector{Blob})
  etc = get_elem(state.etc)
  alpha = one(eltype(inputs[1]))
  beta = zero(eltype(inputs[1]))
  for i = 1:length(inputs)
    CuDNN.softmax_forward(get_cudnn_ctx(backend), CuDNN.CUDNN_SOFTMAX_ACCURATE,
        CuDNN.CUDNN_SOFTMAX_MODE_CHANNEL, alpha, etc.inputs_desc[i], get_ptr(inputs[i]),
        beta, etc.outputs_desc[i], get_ptr(state.blobs[i]))
  end
end

function backward(backend::GPUBackend, state::SoftmaxLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  etc = get_elem(state.etc)
  alpha = one(eltype(inputs[1]))
  beta = zero(eltype(inputs[1]))

  for i = 1:length(inputs)
    diff = diffs[i]
    if !isa(diff, NullBlob)
      CuDNN.softmax_backward(get_cudnn_ctx(backend), CuDNN.CUDNN_SOFTMAX_ACCURATE,
          CuDNN.CUDNN_SOFTMAX_MODE_CHANNEL, alpha, etc.outputs_desc[i], get_ptr(state.blobs[i]),
          etc.outputs_desc[i], get_ptr(state.blobs_diff[i]), beta, etc.inputs_desc[i], get_ptr(diff))
    end
  end
end
