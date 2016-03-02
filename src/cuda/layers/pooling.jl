type CuDNNPoolingState
  pooling_desc :: CuDNN.PoolingDescriptor
  inputs_desc  :: Vector{CuDNN.Tensor4dDescriptor}
  outputs_desc :: Vector{CuDNN.Tensor4dDescriptor}
end

function setup_etc(backend::GPUBackend, layer::PoolingLayer, inputs,
    pooled_width, pooled_height)

  dtype = eltype(inputs[1])

  if isa(layer.pooling, Pooling.Max)
    pooling_mode = CuDNN.CUDNN_POOLING_MAX
  elseif isa(layer.pooling, Pooling.Mean)
    pooling_mode = CuDNN.CUDNN_POOLING_AVERAGE
  else
    error("TODO: pooling mode $(layer.pooling) not supported by CuDNN")
  end
  
  pooling_states = MultiGPUType(backend, CuDNNPoolingState)
  orig_dev = backend.cur_dev.ordinal
  for dev=1:backend.dev_count
      set_dev_id(backend, dev - 1)

      pooling_desc = CuDNN.create_pooling_descriptor(pooling_mode, layer.kernel, layer.stride, layer.pad)
      inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
      outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
    
      for i = 1:length(inputs)
        width,height,channels,num = size(inputs[i])
        inputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype,(width,height,channels,num))
        outputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype,
            (pooled_width[i],pooled_height[i],channels,num))
      end
      pooling_states.elems[dev] = CuDNNPoolingState(pooling_desc, inputs_desc, outputs_desc)
  end
  set_dev_id(backend, orig_dev)
  return pooling_states
end

function shutdown(backend::GPUBackend, state::PoolingLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
  pooling_states = state.etc
  for etc in pooling_states.elems
    CuDNN.destroy_pooling_descriotpr(etc.pooling_desc)
    map(CuDNN.destroy_tensor4d_descriptor, etc.inputs_desc)
    map(CuDNN.destroy_tensor4d_descriptor, etc.outputs_desc)
  end
end

function forward(backend::GPUBackend, state::PoolingLayerState, inputs::Vector{Blob})
  etc = get_elem(state.etc)
  layer = state.layer
  alpha = one(eltype(inputs[1]))
  beta = zero(eltype(inputs[1]))
  for i = 1:length(inputs)
    CuDNN.pooling_forward(get_cudnn_ctx(backend), etc.pooling_desc, alpha,
        etc.inputs_desc[i], get_ptr(inputs[i]), beta,
        etc.outputs_desc[i], get_ptr(state.blobs[i]))
  end
end

function backward(backend::GPUBackend, state::PoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  etc = get_elem(state.etc)
  layer = state.layer
  alpha = one(eltype(inputs[1]))
  beta = zero(eltype(inputs[1]))
  for i = 1:length(inputs)
    if isa(diffs[i], CuTensorBlob)
      CuDNN.pooling_backward(get_cudnn_ctx(backend), etc.pooling_desc, alpha,
          etc.outputs_desc[i], get_ptr(state.blobs[i]),
          etc.outputs_desc[i], get_ptr(state.blobs_diff[i]),
          etc.inputs_desc[i], get_ptr(inputs[i]),
          beta, etc.inputs_desc[i], get_ptr(diffs[i]))
    end
  end
end

