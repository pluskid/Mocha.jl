type CuDNNSoftmaxState
  inputs_desc  :: Vector{CuDNN.Tensor4dDescriptor}
  outputs_desc :: Vector{CuDNN.Tensor4dDescriptor}
end

function setup_etc(sys::System{CuDNNBackend}, layer::SoftmaxLayer, data_type, inputs)
  inputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
  outputs_desc = Array(CuDNN.Tensor4dDescriptor, length(inputs))
  for i = 1:length(inputs)
    inputs_desc[i] = CuDNN.create_tensor4d_descriptor(data_type, size(inputs[i]))
    outputs_desc[i] = CuDNN.create_tensor4d_descriptor(data_type, size(inputs[i]))
  end
  etc = CuDNNSoftmaxState(inputs_desc, outputs_desc)
  return etc
end
function shutdown(sys::System{CuDNNBackend}, state::SoftmaxLayerState)
  map(destroy, state.blobs)
  map(CuDNN.destroy_tensor4d_descriptor, state.etc.inputs_desc)
  map(CuDNN.destroy_tensor4d_descriptor, state.etc.outputs_desc)
end

function forward(sys::System{CuDNNBackend}, state::SoftmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    CuDNN.softmax_forward(sys.backend.cudnn_ctx, CuDNN.CUDNN_SOFTMAX_ACCURATE,
        CuDNN.CUDNN_SOFTMAX_MODE_CHANNEL, state.etc.inputs_desc[i], inputs[i].ptr,
        state.etc.outputs_desc[i], state.blobs[i].ptr)
  end
end

