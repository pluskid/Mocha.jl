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
  pooling_desc = CuDNN.create_pooling_descriptor(pooling_mode, layer.kernel, layer.stride, layer.pad)
  inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
  outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))

  for i = 1:length(inputs)
    width,height,channels,num = size(inputs[i])
    inputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype,(width,height,channels,num))
    outputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype,
        (pooled_width[i],pooled_height[i],channels,num))
  end
  etc = CuDNNPoolingState(pooling_desc, inputs_desc, outputs_desc)
  return etc
end

function shutdown(backend::GPUBackend, state::PoolingLayerState)
  map(destroy, state.blobs)
  map(destroy, state.blobs_diff)
  CuDNN.destroy_pooling_descriotpr(state.etc.pooling_desc)
  map(CuDNN.destroy_tensor4d_descriptor, state.etc.inputs_desc)
  map(CuDNN.destroy_tensor4d_descriptor, state.etc.outputs_desc)
end

function forward(backend::GPUBackend, state::PoolingLayerState, inputs::Vector{Blob})
  layer = state.layer
  alpha = one(eltype(inputs[1]))
  beta = zero(eltype(inputs[1]))
  for i = 1:length(inputs)
    CuDNN.pooling_forward(backend.cudnn_ctx, state.etc.pooling_desc, alpha,
        state.etc.inputs_desc[i], inputs[i].ptr, beta,
        state.etc.outputs_desc[i], state.blobs[i].ptr)
  end
end

function backward(backend::GPUBackend, state::PoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  layer = state.layer
  alpha = one(eltype(inputs[1]))
  beta = zero(eltype(inputs[1]))
  for i = 1:length(inputs)
    if isa(diffs[i], CuTensorBlob)
      CuDNN.pooling_backward(backend.cudnn_ctx, state.etc.pooling_desc, alpha,
          state.etc.outputs_desc[i], state.blobs[i].ptr,
          state.etc.outputs_desc[i], state.blobs_diff[i].ptr,
          state.etc.inputs_desc[i], inputs[i].ptr,
          beta, state.etc.inputs_desc[i], diffs[i].ptr)
    end
  end
end

