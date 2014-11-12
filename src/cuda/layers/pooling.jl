type CuDNNPoolingState
  pooling_desc :: CuDNN.PoolingDescriptor
  inputs_desc  :: Vector{CuDNN.Tensor4dDescriptor}
  outputs_desc :: Vector{CuDNN.Tensor4dDescriptor}
end

function setup_etc(sys::System{CuDNNBackend}, layer::PoolingLayer, inputs,
    pooled_width, pooled_height)

  dtype = eltype(inputs[1])
  width = get_width(inputs[1])
  height = get_height(inputs[1])

  if layer.pad[1] == 0 && layer.pad[2] == 0
    if isa(layer.pooling, Pooling.Max)
      pooling_mode = CuDNN.CUDNN_POOLING_MAX
    elseif isa(layer.pooling, Pooling.Mean)
      pooling_mode = CuDNN.CUDNN_POOLING_AVERAGE
    else
      error("TODO: pooling mode $(layer.pooling) not supported by CuDNN")
    end

    pooling_desc = CuDNN.create_pooling_descriptor(pooling_mode, layer.kernel, layer.stride)
    inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
    outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
    for i = 1:length(inputs)
      inputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype,
          (width,height,get_chann(inputs[i]),get_num(inputs[i])))
      outputs_desc[i] = CuDNN.create_tensor4d_descriptor(dtype,
          (pooled_width,pooled_height,get_chann(inputs[i]),get_num(inputs[i])))
    end
    etc = CuDNNPoolingState(pooling_desc, inputs_desc, outputs_desc)
  else
    error("TODO: CuDNN does not support pooling with padding")
  end
  return etc
end

function forward(sys::System{CuDNNBackend}, state::PoolingLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    CuDNN.pooling_forward(sys.backend.cudnn_ctx, state.etc.pooling_desc,
        state.etc.inputs_desc[i], inputs[i].ptr,
        state.etc.outputs_desc[i], state.blobs[i].ptr)
  end
end

function backward(sys::System{CuDNNBackend}, state::PoolingLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  for i = 1:length(inputs)
    if isa(diffs[i], CuTensorBlob)
      CuDNN.pooling_backward(sys.backend.cudnn_ctx, state.etc.pooling_desc,
          state.etc.outputs_desc[i], state.blobs[i].ptr,
          state.etc.outputs_desc[i], state.blobs_diff[i].ptr,
          state.etc.inputs_desc[i], inputs[i].ptr,
          state.etc.inputs_desc[i], diffs[i].ptr)
    end
  end
end

