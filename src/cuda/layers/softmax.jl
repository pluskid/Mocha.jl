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

#--------------------------------------------------------------------------------
# I'm commenting out this function as currently back propagation for CPU
# mode is not implemented. A logistic-regression like multiclass classification
# top layer should be trained by the SoftmaxLossLayer directly, instead
# of a SoftmaxLayer plus a Multiclass Logistic Loss layer. Because the
# former has much cheaper computation and beter robustness when computing
# the gradients.
#
# If somehow this function is needed in the future, e.g. training a softmax
# layer with some other loss function, blobs_diff field needs to be added
# to SoftmaxLayerState before this function could work properly.
#--------------------------------------------------------------------------------
#function backward(sys::System{CuDNNBackend}, state::SoftmaxLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
#  for i = 1:length(inputs)
#    diff = diffs[i]
#    if isa(diff, CuTensorBlob)
#      CuDNN.softmax_backward(sys.backend.cudnn_ctx, CuDNN.CUDNN_SOFTMAX_ACCURATE,
#          CuDNN.CUDNN_SOFTMAX_MODE_CHANNEL, state.etc.outputs_desc[i], state.blobs[i].ptr,
#          state.etc.outputs_desc[i], state.blobs_diff[i].ptr,
#          state.etc.inputs_desc[i], diffs[i].ptr)
#    end
#  end
#end
